import json
import os
import tempfile
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid, save_image

cache_root = Path(tempfile.gettempdir()) / "cifar_inception_ratio_eval_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.jeffrey.inception_ratio import InceptionRatioPotential, compute_ratio_logits
from src.utils import find_latest_sample, resolve_device, resolve_path


def load_sample(sample_path: Path | None, samples_dir: Path) -> tuple[dict, Path]:
    if sample_path is None:
        sample_path = find_latest_sample(samples_dir)
    if not sample_path.is_file():
        raise FileNotFoundError(f"No sample file found at {sample_path}.")
    return torch.load(sample_path, map_location="cpu"), sample_path


def output_dir_for_sample(project_root: Path, cfg: DictConfig, sample_path: Path) -> Path:
    output_dir = (
        resolve_path(project_root, str(cfg.eval.output_dir))
        / f"{sample_path.stem}_inception_ratio"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def optional_float(value):
    if value is None:
        return None
    return float(value)


def save_logit_histogram(logits: torch.Tensor, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(
        logits.float().numpy(),
        bins=40,
        density=True,
        alpha=0.65,
        color="#2563eb",
        label="samples",
    )
    ax.axvline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    ax.set_xlabel("ratio classifier logit")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    samples_dir = resolve_path(project_root, str(cfg.sampling.output_dir))
    sample_path = (
        resolve_path(project_root, str(cfg.sampling.sample_path))
        if cfg.sampling.sample_path is not None
        else None
    )

    if str(cfg.jeffrey.feature) != "inception_ratio":
        raise ValueError(
            "eval_inception_ratio requires jeffrey=inception_ratio, "
            f"got feature={cfg.jeffrey.feature!r}."
        )

    payload, sample_path = load_sample(sample_path, samples_dir)
    samples = payload["samples"]
    ratio_classifier_path = resolve_path(project_root, str(cfg.jeffrey.ratio_classifier_path))
    potential = InceptionRatioPotential(
        classifier_path=ratio_classifier_path,
        feature_extractor=str(cfg.jeffrey.feature_extractor),
        feature_layer=str(cfg.jeffrey.feature_layer),
        preprocessing=str(cfg.jeffrey.preprocessing),
        logit_clip=optional_float(cfg.jeffrey.logit_clip),
    ).to(device)

    logits = compute_ratio_logits(
        potential=potential,
        samples=samples,
        device=device,
        batch_size=int(cfg.ratio.batch_size),
    )
    probs = torch.sigmoid(logits)

    output_dir = output_dir_for_sample(project_root, cfg, sample_path)
    preview_count = min(int(cfg.eval.preview_num_samples), samples.shape[0])
    preview = (samples[:preview_count].clamp(-1, 1) + 1) / 2
    grid = make_grid(preview, nrow=int(cfg.eval.preview_nrow))
    save_image(grid, output_dir / "samples_preview.png")
    save_logit_histogram(logits, output_dir / "logit_histogram.png")

    metrics = {
        "sample_file": str(sample_path),
        "sample_type": str(payload.get("sample_type", "unknown")),
        "num_samples": int(samples.shape[0]),
        "ratio_classifier_path": str(ratio_classifier_path),
        "feature_extractor": str(cfg.jeffrey.feature_extractor),
        "feature_layer": str(cfg.jeffrey.feature_layer),
        "preprocessing": str(cfg.jeffrey.preprocessing),
        "logit_clip": optional_float(cfg.jeffrey.logit_clip),
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std(unbiased=logits.numel() > 1)),
        "logit_min": float(logits.min()),
        "logit_max": float(logits.max()),
        "sigmoid_mean": float(probs.mean()),
        "sigmoid_std": float(probs.std(unbiased=probs.numel() > 1)),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"loaded samples from: {sample_path}")
    print(f"loaded ratio classifier from: {ratio_classifier_path}")
    print(f"saved eval outputs to: {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(metrics), resolve=True))


if __name__ == "__main__":
    main()
