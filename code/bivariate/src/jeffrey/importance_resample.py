from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.distributions.targets import GaussianTargetMarginal
from src.jeffrey.update import jeffrey_updated_gaussian_params


def build_target_marginal(target_cfg: DictConfig) -> GaussianTargetMarginal:
    target_type = str(target_cfg.type)
    if target_type != "gaussian":
        raise ValueError(f"Unsupported Jeffrey target type: {target_type}")

    return GaussianTargetMarginal(
        mean=float(target_cfg.mean),
        variance=float(target_cfg.variance),
    )


def resolve_path(project_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


def looks_like_jeffrey_sample(path: Path) -> bool:
    return "jeffrey" in path.stem


def find_latest_model_sample(samples_dir: Path) -> Path:
    candidates = sorted(
        samples_dir.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates = [path for path in candidates if not looks_like_jeffrey_sample(path)]

    if not candidates:
        raise FileNotFoundError(f"No model source samples found in {samples_dir}.")
    return candidates[0]


def load_source_samples(cfg: DictConfig, project_root: Path) -> tuple[dict, Path]:
    source_sample_path = cfg.jeffrey.get("source_sample_path")
    if source_sample_path is not None:
        sample_path = resolve_path(project_root, source_sample_path)
    else:
        samples_dir = project_root / str(cfg.sampling.output_dir)
        sample_path = find_latest_model_sample(samples_dir=samples_dir)

    if looks_like_jeffrey_sample(sample_path):
        raise ValueError(
            "Jeffrey resampling only supports original/model samples as input. "
            f"Got Jeffrey-looking source file: {sample_path}"
        )

    payload = torch.load(sample_path, map_location="cpu")
    if "samples" not in payload:
        raise KeyError(f"Expected {sample_path} to contain a 'samples' tensor.")
    if str(payload.get("sample_type", "model")).startswith("jeffrey"):
        raise ValueError(
            "Jeffrey resampling only supports original/model samples as input. "
            f"Got sample_type={payload.get('sample_type')!r} from {sample_path}"
        )
    return payload, sample_path


def old_marginal_distribution(
    joint_mean,
    joint_covariance,
    updated_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.distributions.Normal:
    mean = torch.as_tensor(joint_mean, dtype=dtype, device=device)
    covariance = torch.as_tensor(joint_covariance, dtype=dtype, device=device)
    return torch.distributions.Normal(
        mean[updated_dim],
        torch.sqrt(covariance[updated_dim, updated_dim]),
    )


def jeffrey_log_weights(
    samples: torch.Tensor,
    joint_mean,
    joint_covariance,
    updated_dim: int,
    target_marginal: GaussianTargetMarginal,
) -> torch.Tensor:
    source_marginal = old_marginal_distribution(
        joint_mean=joint_mean,
        joint_covariance=joint_covariance,
        updated_dim=updated_dim,
        dtype=samples.dtype,
        device=samples.device,
    )
    updated_values = samples[:, updated_dim]
    return target_marginal.log_prob(updated_values) - source_marginal.log_prob(
        updated_values
    )


def normalize_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    return torch.softmax(log_weights, dim=0)


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.sum(weights.square())


def make_output_path(cfg: DictConfig, project_root: Path, source_path: Path) -> Path:
    output_dir = project_root / str(cfg.sampling.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.sampling.output_name is not None:
        output_name = str(cfg.sampling.output_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{source_path.stem}_jeffrey_resampled_{timestamp}.pt"

    if not output_name.endswith(".pt"):
        output_name = f"{output_name}.pt"
    return output_dir / output_name


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not bool(cfg.jeffrey.enabled):
        raise ValueError("Jeffrey resampling requires jeffrey.enabled=true")

    project_root = Path(__file__).resolve().parents[2]
    torch.manual_seed(int(cfg.seed))

    source_payload, source_path = load_source_samples(cfg, project_root)
    source_samples = source_payload["samples"].detach().cpu().to(torch.float32)
    if source_samples.ndim != 2 or source_samples.shape[1] != 2:
        raise ValueError(
            "Jeffrey resampling currently expects source samples with shape "
            f"(N, 2), got {tuple(source_samples.shape)}."
        )

    joint_mean = cfg.dataset.mean
    joint_covariance = cfg.dataset.covariance
    updated_dim = int(cfg.jeffrey.updated_dim)
    if updated_dim not in (0, 1):
        raise ValueError("updated_dim must be 0 or 1 for this bivariate example")

    target_marginal = build_target_marginal(cfg.jeffrey.target)
    updated_mean, updated_covariance = jeffrey_updated_gaussian_params(
        joint_mean=joint_mean,
        joint_covariance=joint_covariance,
        updated_dim=updated_dim,
        target_marginal=target_marginal,
    )

    log_weights = jeffrey_log_weights(
        samples=source_samples,
        joint_mean=joint_mean,
        joint_covariance=joint_covariance,
        updated_dim=updated_dim,
        target_marginal=target_marginal,
    )
    weights = normalize_log_weights(log_weights)
    ess = effective_sample_size(weights)

    num_samples = int(cfg.sampling.num_samples)
    indices = torch.multinomial(weights, num_samples=num_samples, replacement=True)
    resampled = source_samples[indices]

    sample_mean = resampled.mean(dim=0)
    sample_covariance = torch.cov(resampled.T)
    source_sample_mean = source_samples.mean(dim=0)
    source_sample_covariance = torch.cov(source_samples.T)
    output_path = make_output_path(cfg, project_root, source_path)

    result = {
        "samples": resampled,
        "sample_type": "jeffrey_resampled",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "source_sample_path": str(source_path),
        "source_sample_type": source_payload.get("sample_type", "model"),
        "source_marginal": "original dataset marginal",
        "updated_dim": updated_dim,
        "target_marginal": {
            "type": "gaussian",
            "mean": target_marginal.mean,
            "variance": target_marginal.variance,
        },
        "updated_mean": updated_mean,
        "updated_covariance": updated_covariance,
        "source_sample_mean": source_sample_mean,
        "source_sample_covariance": source_sample_covariance,
        "sample_mean": sample_mean,
        "sample_covariance": sample_covariance,
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess / source_samples.shape[0],
        "log_weight_min": log_weights.min(),
        "log_weight_max": log_weights.max(),
    }
    torch.save(result, output_path)

    print(f"Loaded source samples from: {source_path}")
    print(f"Saved Jeffrey-resampled samples to: {output_path}")
    print(f"Source sample type: {result['source_sample_type']}")
    print("Source marginal used in denominator: original dataset marginal")
    print(
        "Effective sample size: "
        f"{float(ess):.1f} / {source_samples.shape[0]} "
        f"({float(ess / source_samples.shape[0]):.3f})"
    )
    print(f"Updated analytic mean: {updated_mean.tolist()}")
    print(f"Updated analytic covariance: {updated_covariance.tolist()}")
    print(f"Source sample mean: {source_sample_mean.tolist()}")
    print(f"Source sample covariance: {source_sample_covariance.tolist()}")
    print(f"Resampled mean: {sample_mean.tolist()}")
    print(f"Resampled covariance: {sample_covariance.tolist()}")


if __name__ == "__main__":
    main()
