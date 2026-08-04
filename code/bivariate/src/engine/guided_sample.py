from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.samplers.reverse_guided import (
    GuidedReverseSDESampler,
    resolve_guidance_scale,
)
from src.distributions.targets import build_target_marginal
from src.jeffrey.update import jeffrey_updated_gaussian_params
from src.utils import (
    find_latest_artifact,
    load_model_state,
    resolve_device,
    resolve_path,
    timestamped_output_path,
)


def make_run_label(guidance_scale: float, guidance_start: float, num_steps: int) -> str:
    return (
        f"scale{guidance_scale:g}_guidance_start_{guidance_start:g}_"
        f"T{num_steps}"
    )


def make_output_path(
    cfg: DictConfig,
    project_root: Path,
    run_name: str,
    run_label: str,
) -> Path:
    return timestamped_output_path(
        output_dir=resolve_path(project_root, str(cfg.sampling.output_dir)),
        output_name=cfg.sampling.output_name,
        default_stem=f"{run_name}_naive_guidance_samples_{run_label}",
        extension=".pt",
    )



@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))

    artifact_dir = project_root / str(cfg.training.artifacts_dir)
    if cfg.sampling.artifact_path is None:
        artifact_path = find_latest_artifact(
            artifact_dir,
            preference=str(cfg.sampling.artifact_preference),
        )
    else:
        artifact_path = resolve_path(project_root, str(cfg.sampling.artifact_path))

    payload = torch.load(artifact_path, map_location=device)
    model = instantiate(cfg.model).to(device)
    model.load_state_dict(load_model_state(payload))
    sde = instantiate(cfg.sde)

    num_samples = int(cfg.sampling.num_samples)
    dim = int(cfg.dataset.dim)
    
    target_marginal = build_target_marginal(cfg.jeffrey.target)

    updated_dim = cfg.jeffrey.updated_dim
    updated_mean, updated_covariance = jeffrey_updated_gaussian_params(
        joint_mean=cfg.dataset.mean,
        joint_covariance=cfg.dataset.covariance,
        updated_dim=updated_dim,
        target_marginal=target_marginal,
    )
    original_mean = cfg.dataset.mean[updated_dim]
    original_var = cfg.dataset.covariance[updated_dim][updated_dim]
    original_marginal = torch.distributions.Normal(original_mean, original_var ** 0.5)

    guidance_scale = resolve_guidance_scale(
        cfg.naive_guidance.guidance_scale,
        cfg.naive_guidance.guidance_coeff,
    )
    guidance_start = float(cfg.naive_guidance.guidance_start)
    num_steps = int(cfg.sampling.num_steps)

    sampler = GuidedReverseSDESampler(
        sde,
        target_marginal,
        original_marginal,
        dim,
        updated_dim,
        num_samples,
        guidance_scale=guidance_scale,
        guidance_start=guidance_start,
    )

    samples = sampler.sample(model, num_steps, device, True, True)
    run_name = str(payload.get("run_name") or artifact_path.stem)
    run_label = make_run_label(guidance_scale, guidance_start, num_steps)
    output_path = make_output_path(cfg, project_root, run_name, run_label)
    result = {
        "samples": samples.detach().cpu(),
        "sample_type": "naive_guidance",
        "run_label": run_label,
        "guidance_scale": guidance_scale,
        "guidance_coeff": guidance_scale,
        "guidance_schedule": "delayed_discrete",
        "guidance_start": guidance_start,
        "num_steps": num_steps,
        "artifact_path": str(artifact_path),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "updated_dim": updated_dim,
        "target_marginal": {
            "type": "gaussian",
            "mean": target_marginal.mean,
            "variance": target_marginal.variance,
        },
        "original_marginal": {
            "type": "gaussian",
            "source": "analytic_dataset",
            "mean": float(original_marginal.mean),
            "variance": float(original_marginal.variance),
        },
        "updated_mean": updated_mean,
        "updated_covariance": updated_covariance,
        "sample_mean": samples.mean(dim=0).detach().cpu(),
        "sample_covariance": torch.cov(samples.T.detach().cpu()),
    }
    torch.save(result, output_path)

    print(f"Loaded artifact: {artifact_path}")
    print(f"Saved samples to: {output_path}")
    print(f"Sample mean: {result['sample_mean'].tolist()}")
    print(f"Sample covariance: {result['sample_covariance'].tolist()}")


if __name__ == "__main__":
    main()
