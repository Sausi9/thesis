from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.preview import make_preview_figure


def resolve_path(project_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


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

    fig, mean, covariance = make_preview_figure(
        samples,
        sample_cfg,
        sample_contours=True,
        title_suffix="eval",
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
