import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid, save_image

cache_root = Path(tempfile.gettempdir()) / "cifar_brightness_naive_sweep_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.distributions.targets import GaussianTargetMarginal, estimate_gaussian_marginal
from src.jeffrey.brightness import brightness
from src.samplers.reverse_guided import GuidedReverseSDESampler
from src.utils import (
    find_latest_artifact,
    find_latest_sample,
    load_model_state,
    resolve_device,
    resolve_path,
)


def load_model_samples(sample_path: Path) -> torch.Tensor:
    payload = torch.load(sample_path, map_location="cpu")
    if payload.get("sample_type") != "model":
        raise ValueError(
            f"Expected unconditional model samples, got {payload.get('sample_type')}"
        )
    return payload["samples"]


def make_sweep_dir(project_root: Path, cfg: DictConfig) -> Path:
    output_dir = resolve_path(project_root, str(cfg.sweep.output_dir))
    output_name = cfg.sweep.output_name
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"brightness_naive_guidance_sweep_{timestamp}"
    sweep_dir = output_dir / str(output_name)
    sweep_dir.mkdir(parents=True, exist_ok=False)
    return sweep_dir


def safe_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def setting_stem(guidance_coeff: float, guidance_start: float) -> str:
    return f"coeff{safe_value(guidance_coeff)}_start{safe_value(guidance_start)}"


def save_preview(samples: torch.Tensor, path: Path, nrow: int, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview_count = min(int(count), int(samples.shape[0]))
    preview = (samples[:preview_count].clamp(-1, 1) + 1) / 2
    grid = make_grid(preview, nrow=int(nrow))
    save_image(grid, path)


def save_results_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], path: Path, original_mean: float, target_mean: float) -> None:
    starts = sorted({float(row["guidance_start"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for guidance_start in starts:
        line_rows = sorted(
            [
                row
                for row in rows
                if float(row["guidance_start"]) == guidance_start
            ],
            key=lambda row: float(row["guidance_coeff"]),
        )
        x = [float(row["guidance_coeff"]) for row in line_rows]
        y = [float(row["brightness_mean"]) for row in line_rows]
        yerr = [float(row["brightness_standard_error"]) for row in line_rows]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=f"start={guidance_start:g}",
        )

    ax.axhline(
        target_mean,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label=f"target mean={target_mean:g}",
    )
    ax.axhline(
        original_mean,
        color="#6b7280",
        linestyle=":",
        linewidth=1.2,
        label=f"model mean={original_mean:.3f}",
    )
    ax.set_xlabel("guidance scale")
    ax.set_ylabel("sample brightness mean")
    ax.set_title("Brightness naive guidance sweep")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))

    backend = str(OmegaConf.select(cfg, "sampling.backend", default="score_sde"))
    if backend != "score_sde":
        raise ValueError(
            "brightness_naive_guidance_sweep is score-SDE only; "
            f"got sampling.backend={backend!r}."
        )

    sweep_dir = make_sweep_dir(project_root, cfg)
    (sweep_dir / "previews").mkdir(parents=True, exist_ok=True)

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
    requested_weight_type = str(OmegaConf.select(cfg, "sampling.weight_type", default="raw"))
    model_state, loaded_weight_type = load_model_state(
        payload,
        weight_type=requested_weight_type,
        return_weight_type=True,
    )
    model.load_state_dict(model_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sde = instantiate(cfg.sde)

    if cfg.jeffrey.source_sample_path is None:
        source_path = find_latest_sample(
            resolve_path(project_root, str(cfg.sampling.output_dir)),
            sample_type="model",
        )
    else:
        source_path = resolve_path(project_root, str(cfg.jeffrey.source_sample_path))

    source_samples = load_model_samples(source_path)
    source_brightness = brightness(source_samples)
    original_marginal = estimate_gaussian_marginal(source_brightness)
    original_mean = float(original_marginal.loc.detach().cpu())
    original_std = float(original_marginal.scale.detach().cpu())

    target_marginal = GaussianTargetMarginal(
        mean=float(cfg.sweep.target_mean),
        variance=float(cfg.sweep.target_variance),
    )

    num_samples_per_setting = int(cfg.sweep.num_samples_per_setting)
    sample_shape = tuple(int(v) for v in cfg.dataset.shape)
    num_steps = int(cfg.sampling.num_steps)
    batch_size = int(cfg.sampling.batch_size)
    max_guidance_grad_norm = cfg.naive_guidance.max_guidance_grad_norm

    rows = []
    for guidance_start in [float(value) for value in cfg.sweep.guidance_starts]:
        for guidance_coeff in [float(value) for value in cfg.sweep.guidance_coeffs]:
            print(
                "Running setting: "
                f"guidance_coeff={guidance_coeff:g}, "
                f"guidance_start={guidance_start:g}, "
                f"samples={num_samples_per_setting}"
            )
            all_samples = []
            remaining_samples = num_samples_per_setting
            while remaining_samples > 0:
                batch_n = min(batch_size, remaining_samples)
                sampler = GuidedReverseSDESampler(
                    sde=sde,
                    target_marginal=target_marginal,
                    original_marginal=original_marginal,
                    num_samples=batch_n,
                    sample_shape=sample_shape,
                    guidance_coeff=guidance_coeff,
                    guidance_start=guidance_start,
                    max_guidance_grad_norm=max_guidance_grad_norm,
                )
                samples_batch = sampler.sample(
                    model,
                    num_steps,
                    device,
                    return_mean=bool(cfg.sampling.return_mean),
                    progress=bool(cfg.sampling.progress),
                )
                all_samples.append(samples_batch.detach().cpu())
                remaining_samples -= batch_n

            samples = torch.cat(all_samples, dim=0)
            values = brightness(samples)
            value_std = values.std(unbiased=values.numel() > 1)
            value_se = value_std / values.numel() ** 0.5
            stem = setting_stem(guidance_coeff, guidance_start)

            sample_path = None
            if bool(cfg.sweep.save_samples):
                sample_dir = sweep_dir / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)
                sample_path = sample_dir / f"{stem}.pt"
                torch.save(
                    {
                        "samples": samples,
                        "sample_type": "brightness_naive_guidance_sweep_setting",
                        "guidance_coeff": guidance_coeff,
                        "guidance_start": guidance_start,
                        "target_marginal": {
                            "feature": "brightness",
                            "type": "gaussian",
                            "mean": target_marginal.mean,
                            "variance": target_marginal.variance,
                            "std": target_marginal.std,
                        },
                        "original_marginal": {
                            "feature": "brightness",
                            "type": "gaussian",
                            "mean": original_mean,
                            "std": original_std,
                        },
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    sample_path,
                )

            preview_path = None
            if bool(cfg.sweep.save_previews):
                preview_path = sweep_dir / "previews" / f"{stem}.png"
                save_preview(
                    samples,
                    preview_path,
                    nrow=int(cfg.sampling.preview_nrow),
                    count=int(cfg.sampling.preview_num_samples),
                )

            row = {
                "guidance_coeff": guidance_coeff,
                "guidance_start": guidance_start,
                "num_samples": int(values.numel()),
                "brightness_mean": float(values.mean()),
                "brightness_std": float(value_std),
                "brightness_standard_error": float(value_se),
                "brightness_min": float(values.min()),
                "brightness_max": float(values.max()),
                "target_mean": float(target_marginal.mean),
                "target_std": float(target_marginal.std),
                "target_variance": float(target_marginal.variance),
                "original_mean": original_mean,
                "original_std": original_std,
                "num_steps": num_steps,
                "batch_size": batch_size,
                "max_guidance_grad_norm": max_guidance_grad_norm,
                "artifact_path": str(artifact_path),
                "source_sample_path": str(source_path),
                "loaded_weight_type": loaded_weight_type,
                "sample_path": "" if sample_path is None else str(sample_path),
                "preview_path": "" if preview_path is None else str(preview_path),
            }
            rows.append(row)
            print(
                f"  mean={row['brightness_mean']:.6f}, "
                f"std={row['brightness_std']:.6f}, "
                f"se={row['brightness_standard_error']:.6f}"
            )

    save_results_csv(rows, sweep_dir / "results.csv")
    metadata = {
        "sweep_dir": str(sweep_dir),
        "artifact_path": str(artifact_path),
        "source_sample_path": str(source_path),
        "requested_weight_type": requested_weight_type,
        "loaded_weight_type": loaded_weight_type,
        "target_mean": float(target_marginal.mean),
        "target_variance": float(target_marginal.variance),
        "original_mean": original_mean,
        "original_std": original_std,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "results": rows,
    }
    with (sweep_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    plot_path = sweep_dir / "brightness_mean_vs_guidance_coeff.png"
    save_plot(
        rows,
        plot_path,
        original_mean=original_mean,
        target_mean=float(target_marginal.mean),
    )

    print(f"Saved sweep outputs to: {sweep_dir}")
    print(f"Saved results CSV to: {sweep_dir / 'results.csv'}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
