from pathlib import Path
from textwrap import shorten

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


def load_sample(sample_path: Path | None, run_name: str | None, samples_dir: Path) -> tuple[dict, Path]:
    if sample_path is None:
        return load_latest_sample(run_name, samples_dir)

    if not sample_path.is_file():
        raise FileNotFoundError(f"No sample file found at {sample_path}.")

    payload = torch.load(sample_path, map_location="cpu")
    return payload, sample_path


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


def get_nested(mapping, keys: tuple[str, ...], default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def build_run_label(payload: dict, sample_cfg: DictConfig) -> str:
    if "run_label" in payload:
        return str(payload["run_label"])

    sample_type = str(payload.get("sample_type", "unknown"))
    if sample_type != "tds":
        return sample_type

    cfg_dict = OmegaConf.to_container(sample_cfg, resolve=True)
    twist_type = get_nested(cfg_dict, ("sampler", "twist_type"), "unknown")
    guidance_ramp = get_nested(cfg_dict, ("sampler", "guidance_ramp"), "none")
    resample_type = get_nested(cfg_dict, ("sampler", "resample_type"), "unknown")
    adaptive_resampling = get_nested(cfg_dict, ("sampler", "adaptive_resampling"), "unknown")
    ess_threshold = get_nested(cfg_dict, ("sampler", "ess_threshold"), "unknown")
    num_particles = get_nested(cfg_dict, ("sampler", "num_particles"), "?")
    num_steps = get_nested(cfg_dict, ("sampling", "num_steps"), "?")
    mode = "adaptive" if adaptive_resampling is True else "always"
    threshold = f"_ess{ess_threshold:g}" if adaptive_resampling is True else ""
    ramp = "" if guidance_ramp == "none" else f"_{guidance_ramp}-ramp"
    return f"{twist_type}{ramp}_{resample_type}_{mode}{threshold}_K{num_particles}_T{num_steps}"


def build_info_lines(payload: dict, sample_cfg: DictConfig, sample_path: Path) -> list[str]:
    cfg_dict = OmegaConf.to_container(sample_cfg, resolve=True)
    sample_type = str(payload.get("sample_type", "unknown"))
    lines = [
        f"type = {sample_type}",
        f"file = {shorten(sample_path.name, width=26, placeholder='...')}",
    ]

    if sample_type == "tds":
        twist_type = payload.get(
            "twist_type",
            get_nested(cfg_dict, ("sampler", "twist_type"), "unknown"),
        )
        guidance_ramp = payload.get(
            "guidance_ramp",
            get_nested(cfg_dict, ("sampler", "guidance_ramp"), "none"),
        )
        resample_type = payload.get(
            "resample_type",
            get_nested(cfg_dict, ("sampler", "resample_type"), "unknown"),
        )

        adaptive_resampling = payload.get(
            "adaptive_resampling",
            get_nested(cfg_dict, ("sampler", "adaptive_resampling"), "unknown"),
        )

        ess_threshold = payload.get(
            "ess_threshold",
            get_nested(cfg_dict, ("sampler", "ess_threshold"), "unknown"),
        )
        num_particles = payload.get(
            "num_particles",
            get_nested(cfg_dict, ("sampler", "num_particles"), "?"),
        )
        num_steps = payload.get(
            "num_steps",
            get_nested(cfg_dict, ("sampling", "num_steps"), "?"),
        )
        seed = payload.get("seed", get_nested(cfg_dict, ("seed",), "?"))
        lines.extend(
            [
                f"twist = {twist_type}",
                f"guidance ramp = {guidance_ramp}",
                f"resample = {resample_type}",
                f"adaptive resampling = {adaptive_resampling}",
                f"ess threshold = {ess_threshold}",
                f"K = {num_particles}",
                f"steps = {num_steps}",
                f"seed = {seed}",
            ]
        )

    return lines


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    samples_dir = resolve_path(project_root, str(cfg.sampling.output_dir))
    sample_path = (
        resolve_path(project_root, str(cfg.sampling.sample_path))
        if cfg.sampling.sample_path is not None
        else None
    )
    run_name = str(cfg.run_name) if cfg.run_name is not None else None

    payload, sample_path = load_sample(sample_path, run_name, samples_dir)
    samples = payload["samples"]
    current_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    saved_cfg = OmegaConf.create(payload.get("config", {}))
    sample_cfg = OmegaConf.merge(current_cfg, saved_cfg)

    sample_type = str(payload.get("sample_type", "unknown"))
    run_label = build_run_label(payload, sample_cfg)
    fig, mean, covariance = make_preview_figure(
        samples,
        sample_cfg,
        sample_contours=True,
        title_suffix=run_label,
        extra_contours=build_extra_contours(payload),
        info_lines=build_info_lines(payload, sample_cfg, sample_path),
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
