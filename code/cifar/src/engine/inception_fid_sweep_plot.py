import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

cache_root = Path(tempfile.gettempdir()) / "cifar_inception_fid_sweep_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import resolve_path


def safe_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def sample_name(template: str, guidance_start: float, guidance_scale: float) -> str:
    return template.format(
        start=safe_value(guidance_start),
        scale=safe_value(guidance_scale),
    )


def make_sweep_dir(project_root: Path, cfg: DictConfig) -> Path:
    output_dir = resolve_path(project_root, str(cfg.sweep.output_dir))
    output_name = cfg.sweep.output_name
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"inception_tds_fid_sweep_{timestamp}"
    sweep_dir = output_dir / str(output_name)
    sweep_dir.mkdir(parents=True, exist_ok=False)
    return sweep_dir


def load_metrics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No metrics.json found at {path}.")
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if "fid" not in metrics:
        raise KeyError(f"Metrics file {path} does not contain 'fid'.")
    return metrics


def save_results_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], path: Path, unguided_fid: float | None) -> None:
    starts = sorted({float(row["guidance_start"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for guidance_start in starts:
        line_rows = sorted(
            [
                row
                for row in rows
                if float(row["guidance_start"]) == guidance_start
            ],
            key=lambda row: float(row["guidance_scale"]),
        )
        x = [float(row["guidance_scale"]) for row in line_rows]
        y = [float(row["fid"]) for row in line_rows]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            label=f"start={guidance_start:g}",
        )

    if unguided_fid is not None:
        ax.axhline(
            float(unguided_fid),
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            label=f"unguided FID={float(unguided_fid):g}",
        )

    ax.set_xlabel("guidance scale")
    ax.set_ylabel("FID")
    ax.set_title("Inception TDS FID sweep")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    sweep_dir = make_sweep_dir(project_root, cfg)
    eval_dir = resolve_path(project_root, str(cfg.sweep.eval_dir))
    template = str(cfg.sweep.sample_name_template)

    rows = []
    for guidance_start in [float(value) for value in cfg.sweep.guidance_starts]:
        for guidance_scale in [float(value) for value in cfg.sweep.guidance_scales]:
            name = sample_name(template, guidance_start, guidance_scale)
            metrics_path = eval_dir / name / "metrics.json"
            metrics = load_metrics(metrics_path)
            rows.append(
                {
                    "guidance_start": guidance_start,
                    "guidance_scale": guidance_scale,
                    "sample_name": name,
                    "metrics_path": str(metrics_path),
                    "fid": float(metrics["fid"]),
                    "fid_num_samples": int(metrics.get("fid_num_samples", 0)),
                    "sample_type": str(metrics.get("sample_type", "unknown")),
                }
            )

    save_results_csv(rows, sweep_dir / "results.csv")
    metadata = {
        "sweep_dir": str(sweep_dir),
        "config": OmegaConf.to_container(cfg.sweep, resolve=True),
        "results": rows,
    }
    with (sweep_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    unguided_fid = cfg.sweep.unguided_fid
    if unguided_fid is not None:
        unguided_fid = float(unguided_fid)
    plot_path = sweep_dir / "fid_vs_guidance_scale.png"
    save_plot(rows, plot_path, unguided_fid=unguided_fid)

    print(f"Saved sweep outputs to: {sweep_dir}")
    print(f"Saved results CSV to: {sweep_dir / 'results.csv'}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
