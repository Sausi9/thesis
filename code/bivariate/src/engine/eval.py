from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.preview import make_preview_figure
from src.utils import resolve_path


def load_latest_sample(run_name: str | None, samples_dir: Path) -> tuple[dict, Path]:
    pattern = f"{run_name}*_samples_*.pt" if run_name else "*.pt"
    files = sorted(
        samples_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No samples matching {pattern} in {samples_dir}.")

    path = files[0]
    payload = torch.load(path, map_location="cpu")
    return payload, path


def make_output_path(sample_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{sample_path.stem}_eval.png"


def build_extra_contours(payload: dict) -> list[dict]:
    if "updated_mean" not in payload or "updated_covariance" not in payload:
        return []
    return [
        {
            "mean": payload["updated_mean"],
            "covariance": payload["updated_covariance"],
            "label": "exact Jeffrey",
            "color": "#7c3aed",
            "linestyle": "-.",
        }
    ]


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    samples_dir = resolve_path(project_root, str(cfg.sampling.output_dir))
    run_name = str(cfg.run_name) if cfg.run_name is not None else None

    payload, sample_path = load_latest_sample(run_name, samples_dir)
    samples = payload["samples"]
    current_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    saved_cfg = OmegaConf.create(payload.get("config", {}))
    sample_cfg = OmegaConf.merge(current_cfg, saved_cfg)

    sample_type = str(payload.get("sample_type", "unknown"))
    fig, mean, covariance = make_preview_figure(
        samples,
        sample_cfg,
        sample_contours=True,
        title_suffix=sample_type,
        extra_contours=build_extra_contours(payload),
    )
    output_path = make_output_path(sample_path, project_root / "runs/evals")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"loaded samples from: {sample_path}")
    print(f"saved eval figure to: {output_path}")
    print(f"sample mean: {mean.tolist()}")
    print(f"sample covariance: {covariance.tolist()}")


if __name__ == "__main__":
    main()
