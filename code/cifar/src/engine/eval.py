import json
import os
import tempfile
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid, save_image

cache_root = Path(tempfile.gettempdir()) / "cifar_eval_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from src.distributions.targets import (
    GaussianTargetMarginal,
    build_target_marginal,
    estimate_gaussian_marginal,
)
from src.jeffrey.brightness import brightness
from src.utils import find_latest_sample, resolve_path


class GeneratedSamplesDataset(Dataset):
    def __init__(self, samples: torch.Tensor):
        if samples.ndim != 4:
            raise ValueError(
                f"Expected generated samples with shape [N, C, H, W], got {tuple(samples.shape)}."
            )
        if samples.shape[1:] != (3, 32, 32):
            raise ValueError(
                "FID currently expects CIFAR-shaped samples [N, 3, 32, 32], "
                f"got {tuple(samples.shape)}."
            )
        self.samples = ((samples.detach().cpu().clamp(-1, 1) + 1) * 127.5).round()
        self.samples = self.samples.clamp(0, 255).to(torch.uint8)

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


def load_sample(sample_path: Path | None, samples_dir: Path) -> tuple[dict, Path]:
    if sample_path is None:
        sample_path = find_latest_sample(samples_dir)
    if not sample_path.is_file():
        raise FileNotFoundError(f"No sample file found at {sample_path}.")
    return torch.load(sample_path, map_location="cpu"), sample_path


def target_from_payload_or_cfg(payload: dict, cfg: DictConfig):
    target_payload = payload.get("target_marginal")
    if isinstance(target_payload, dict) and target_payload.get("type") == "gaussian":
        return GaussianTargetMarginal(
            mean=float(target_payload["mean"]),
            variance=float(target_payload["variance"]),
        )
    return build_target_marginal(cfg.jeffrey.target)


def output_dir_for_sample(project_root: Path, cfg: DictConfig, sample_path: Path) -> Path:
    output_dir = resolve_path(project_root, str(cfg.eval.output_dir)) / sample_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def histogram_metrics(values: torch.Tensor, target, bins: int, value_range: tuple[float, float]):
    min_value, max_value = value_range
    counts = torch.histc(values.float(), bins=bins, min=min_value, max=max_value)
    sample_prob = counts / counts.sum().clamp_min(1.0)

    edges = torch.linspace(min_value, max_value, bins + 1)
    target_dist = torch.distributions.Normal(
        torch.tensor(float(target.mean)),
        torch.tensor(float(target.std)),
    )
    target_prob = target_dist.cdf(edges[1:]) - target_dist.cdf(edges[:-1])
    target_prob = target_prob / target_prob.sum().clamp_min(1e-12)

    eps = 1e-12
    kl = (sample_prob * ((sample_prob + eps) / (target_prob + eps)).log()).sum()
    tv = 0.5 * (sample_prob - target_prob).abs().sum()
    width = (max_value - min_value) / bins
    w1 = (sample_prob.cumsum(0) - target_prob.cumsum(0)).abs().sum() * width
    return {
        "histogram_kl_sample_to_target": float(kl),
        "histogram_tv": float(tv),
        "histogram_w1": float(w1),
    }


def normal_pdf(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    dist = torch.distributions.Normal(torch.tensor(mean), torch.tensor(std))
    return torch.exp(dist.log_prob(x))


def save_brightness_plot(
    values: torch.Tensor,
    target,
    original,
    output_path: Path,
    bins: int,
    value_range: tuple[float, float],
) -> None:
    x = torch.linspace(value_range[0], value_range[1], 400)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(
        values.numpy(),
        bins=bins,
        range=value_range,
        density=True,
        alpha=0.55,
        color="#2563eb",
        label="samples",
    )
    ax.plot(
        x.numpy(),
        normal_pdf(x, target.mean, target.std).numpy(),
        color="#7c3aed",
        linewidth=2.0,
        label="target",
    )
    if original is not None:
        ax.plot(
            x.numpy(),
            normal_pdf(x, float(original.loc), float(original.scale)).numpy(),
            color="#f97316",
            linewidth=2.0,
            linestyle="--",
            label="original estimate",
        )
    ax.set_xlabel("brightness")
    ax.set_ylabel("density")
    ax.set_xlim(*value_range)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def resolve_fid_cuda(cfg: DictConfig) -> bool:
    cuda = str(cfg.eval.fid.cuda)
    if cuda == "auto":
        return torch.cuda.is_available()
    if cuda.lower() in {"true", "1", "yes"}:
        return True
    if cuda.lower() in {"false", "0", "no"}:
        return False
    raise ValueError("eval.fid.cuda must be one of: auto, true, false.")


def compute_fid_metrics(
    *,
    samples: torch.Tensor,
    sample_path: Path,
    project_root: Path,
    cfg: DictConfig,
) -> dict:
    from torch_fidelity import calculate_metrics

    fid_cfg = cfg.eval.fid
    requested_num_samples = int(fid_cfg.num_samples)
    min_samples = int(fid_cfg.min_samples)
    if requested_num_samples <= 1:
        raise ValueError("eval.fid.num_samples must be greater than 1.")
    if min_samples <= 1:
        raise ValueError("eval.fid.min_samples must be greater than 1.")

    available_samples = int(samples.shape[0])
    fid_num_samples = min(requested_num_samples, available_samples)
    if fid_num_samples < min_samples:
        raise ValueError(
            "Not enough generated samples for configured FID: "
            f"available={available_samples}, requested={requested_num_samples}, "
            f"min_required={min_samples}. Generate more samples or lower eval.fid.min_samples."
        )

    generated_dataset = GeneratedSamplesDataset(samples[:fid_num_samples])
    cache_root = resolve_path(project_root, str(fid_cfg.cache_root))
    cache_root.mkdir(parents=True, exist_ok=True)
    datasets_root = resolve_path(project_root, str(fid_cfg.datasets_root))
    input_cache_name = fid_cfg.input_cache_name
    if input_cache_name is None:
        input_cache_name = f"{sample_path.stem}-{fid_num_samples}"

    raw_metrics = calculate_metrics(
        input1=generated_dataset,
        input2=str(fid_cfg.reference),
        cuda=resolve_fid_cuda(cfg),
        batch_size=int(fid_cfg.batch_size),
        fid=True,
        cache_root=str(cache_root),
        cache=True,
        input1_cache_name=str(input_cache_name),
        datasets_root=str(datasets_root),
        datasets_download=bool(fid_cfg.datasets_download),
        feature_extractor=str(fid_cfg.feature_extractor),
        save_cpu_ram=bool(fid_cfg.save_cpu_ram),
        verbose=bool(fid_cfg.verbose),
    )

    fid = float(raw_metrics["frechet_inception_distance"])
    return {
        "fid": fid,
        "fid_num_samples": fid_num_samples,
        "fid_reference": str(fid_cfg.reference),
        "fid_feature_extractor": str(fid_cfg.feature_extractor),
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    samples_dir = resolve_path(project_root, str(cfg.sampling.output_dir))
    sample_path = (
        resolve_path(project_root, str(cfg.sampling.sample_path))
        if cfg.sampling.sample_path is not None
        else None
    )

    payload, sample_path = load_sample(sample_path, samples_dir)
    samples = payload["samples"]
    values = brightness(samples)
    target = target_from_payload_or_cfg(payload, cfg)

    original_payload = payload.get("original_marginal")
    original = None
    if original_payload is not None:
        original = torch.distributions.Normal(
            torch.tensor(float(original_payload["mean"])),
            torch.tensor(float(original_payload["std"])),
        )
    elif payload.get("sample_type") == "model":
        original = estimate_gaussian_marginal(values)

    value_range = tuple(float(v) for v in cfg.eval.brightness_range)
    bins = int(cfg.eval.bins)
    metrics = {
        "sample_file": str(sample_path),
        "sample_type": str(payload.get("sample_type", "unknown")),
        "num_samples": int(samples.shape[0]),
        "brightness_mean": float(values.mean()),
        "brightness_std": float(values.std(unbiased=values.numel() > 1)),
        "target_mean": float(target.mean),
        "target_std": float(target.std),
        "target_variance": float(target.variance),
    }
    if original is not None:
        metrics["original_mean"] = float(original.loc)
        metrics["original_std"] = float(original.scale)
    metrics.update(histogram_metrics(values, target, bins=bins, value_range=value_range))
    if bool(cfg.eval.fid.enabled):
        metrics.update(
            compute_fid_metrics(
                samples=samples,
                sample_path=sample_path,
                project_root=project_root,
                cfg=cfg,
            )
        )

    output_dir = output_dir_for_sample(project_root, cfg, sample_path)
    preview_count = min(int(cfg.eval.preview_num_samples), samples.shape[0])
    preview = (samples[:preview_count].clamp(-1, 1) + 1) / 2
    grid = make_grid(preview, nrow=int(cfg.eval.preview_nrow))
    save_image(grid, output_dir / "samples_preview.png")
    save_brightness_plot(
        values,
        target,
        original,
        output_dir / "brightness_histogram.png",
        bins=bins,
        value_range=value_range,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"loaded samples from: {sample_path}")
    print(f"saved eval outputs to: {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(metrics), resolve=True))


if __name__ == "__main__":
    main()
