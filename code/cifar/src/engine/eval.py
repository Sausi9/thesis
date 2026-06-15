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

from src.distributions.targets import (
    GaussianTargetMarginal,
    build_target_marginal,
    estimate_gaussian_marginal,
)
from src.jeffrey.brightness import brightness
from src.utils import find_latest_sample, resolve_path


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
