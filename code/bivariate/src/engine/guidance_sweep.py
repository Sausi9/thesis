import csv
import json
import os
import tempfile
from datetime import datetime
from itertools import product
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

cache_root = Path(tempfile.gettempdir()) / "bivariate_guidance_sweep_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.preview import plot_gaussian_contours
from src.distributions.gaussian import estimate_model_gaussian_params
from src.distributions.targets import GaussianTargetMarginal, estimate_model_marginal
from src.engine.tds_sample import make_run_label as make_tds_run_label
from src.jeffrey.update import jeffrey_updated_gaussian_params
from src.samplers.reverse_guided import GuidedReverseSDESampler
from src.samplers.tds import TDSSampler
from src.utils import load_model_state, resolve_device, resolve_path


def safe_value(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def setting_id(method: str, target_mean: float, scale: float, start: float) -> str:
    method_name = "naive" if method == "naive_guidance" else "tds"
    return (
        f"{method_name}_target{safe_value(target_mean)}_"
        f"scale{safe_value(scale)}_start{safe_value(start)}"
    )


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def atomic_csv_save(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def psd_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    return eigenvectors @ torch.diag(eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T


def gaussian_w2_squared(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    target_mean: torch.Tensor,
    target_covariance: torch.Tensor,
) -> float:
    mean = mean.double()
    covariance = covariance.double()
    target_mean = target_mean.double()
    target_covariance = target_covariance.double()
    target_sqrt = psd_sqrt(target_covariance)
    middle_sqrt = psd_sqrt(target_sqrt @ covariance @ target_sqrt)
    distance = (mean - target_mean).square().sum() + torch.trace(
        covariance + target_covariance - 2.0 * middle_sqrt
    )
    return float(distance.clamp_min(0).cpu())


def sample_metrics(
    samples: torch.Tensor,
    updated_dim: int,
    target_mean: torch.Tensor,
    target_covariance: torch.Tensor,
) -> dict:
    samples = samples.detach().cpu().float()
    sample_mean = samples.mean(dim=0)
    sample_covariance = torch.cov(samples.T)
    sample_std = sample_covariance[updated_dim, updated_dim].clamp_min(0).sqrt()
    target_std = target_covariance[updated_dim, updated_dim].sqrt()
    marginal_w2_squared = (
        (sample_mean[updated_dim] - target_mean[updated_dim]).square()
        + (sample_std - target_std).square()
    )
    return {
        "sample_mean": sample_mean,
        "sample_covariance": sample_covariance,
        "updated_marginal_mean": float(sample_mean[updated_dim]),
        "updated_marginal_std": float(sample_std),
        "marginal_w2_squared": float(marginal_w2_squared),
        "joint_w2_squared": gaussian_w2_squared(
            sample_mean,
            sample_covariance,
            target_mean,
            target_covariance,
        ),
        "mean_error_l2": float(torch.linalg.vector_norm(sample_mean - target_mean)),
        "covariance_error_fro": float(
            torch.linalg.matrix_norm(sample_covariance - target_covariance)
        ),
    }


def load_source_samples(source_path: Path, artifact_path: Path) -> tuple[torch.Tensor, dict]:
    payload = torch.load(source_path, map_location="cpu")
    if payload.get("sample_type") != "model":
        raise ValueError(
            f"Expected unconditional sample_type='model', got {payload.get('sample_type')!r}."
        )
    recorded_artifact = payload.get("artifact_path")
    if recorded_artifact is None:
        raise ValueError("The source sample payload does not record artifact_path.")
    if Path(str(recorded_artifact)).name != artifact_path.name:
        raise ValueError(
            "The unconditional source was produced by a different model artifact: "
            f"source={Path(str(recorded_artifact)).name!r}, "
            f"selected={artifact_path.name!r}."
        )
    return payload["samples"], payload


def make_sweep_dir(project_root: Path, cfg: DictConfig) -> Path:
    output_root = resolve_path(project_root, str(cfg.sweep.output_dir))
    output_name = cfg.sweep.output_name
    if output_name is None:
        if bool(cfg.sweep.resume):
            raise ValueError("sweep.output_name must be fixed when sweep.resume=true.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{cfg.sweep.stage}_{timestamp}"
    sweep_dir = output_root / str(output_name)
    sweep_dir.mkdir(parents=True, exist_ok=bool(cfg.sweep.resume))
    return sweep_dir


def stage_budget(cfg: DictConfig) -> tuple[int, int, int]:
    stage = str(cfg.sweep.stage)
    if stage not in ("calibration", "confirmation"):
        raise ValueError(f"Unsupported sweep.stage={stage!r}.")
    stage_cfg = cfg.sweep[stage]
    return (
        int(stage_cfg.num_samples_per_setting),
        int(stage_cfg.num_particles),
        int(stage_cfg.seed),
    )


def compatibility_metadata(
    cfg: DictConfig,
    artifact_path: Path,
    source_path: Path,
) -> dict:
    num_samples, num_particles, seed = stage_budget(cfg)
    stage = str(cfg.sweep.stage)
    metadata = {
        "stage": stage,
        "artifact_path": str(artifact_path),
        "source_sample_path": str(source_path),
        "methods": [str(value) for value in cfg.sweep.methods],
        "target_means": [float(value) for value in cfg.sweep.target_means],
        "target_variance": float(cfg.sweep.target_variance),
        "guidance_scales": [float(value) for value in cfg.sweep.guidance_scales],
        "guidance_starts": [float(value) for value in cfg.sweep.guidance_starts],
        "num_samples_per_setting": num_samples,
        "num_particles": num_particles,
        "seed": seed,
        "num_steps": int(cfg.sweep.num_steps),
        "batch_size": int(cfg.sweep.batch_size),
        "updated_dim": int(cfg.jeffrey.updated_dim),
        "naive": OmegaConf.to_container(cfg.sweep.naive, resolve=True),
        "tds": OmegaConf.to_container(cfg.sweep.tds, resolve=True),
    }
    if stage == "confirmation":
        if cfg.sweep.calibration_results_path is None:
            raise ValueError(
                "sweep.calibration_results_path is required for confirmation."
            )
        metadata["calibration_results_path"] = str(
            resolve_path(Path(__file__).resolve().parents[2], str(cfg.sweep.calibration_results_path))
        )
    return metadata


def make_calibration_settings(cfg: DictConfig) -> list[dict]:
    methods = [str(value) for value in cfg.sweep.methods]
    expected_methods = {"naive_guidance", "tds"}
    unknown = set(methods) - expected_methods
    if unknown:
        raise ValueError(f"Unsupported sweep methods: {sorted(unknown)}")
    return [
        {
            "method": method,
            "target_mean": float(target_mean),
            "guidance_scale": float(scale),
            "guidance_start": float(start),
            "selected_from": "",
        }
        for method, target_mean, start, scale in product(
            methods,
            cfg.sweep.target_means,
            cfg.sweep.guidance_starts,
            cfg.sweep.guidance_scales,
        )
    ]


def winner_key(row: dict) -> tuple:
    scale = float(row["guidance_scale"])
    return (
        float(row["marginal_w2_squared"]),
        float(row["joint_w2_squared"]),
        abs(scale - 1.0),
        scale,
        float(row["guidance_start"]),
    )


def make_confirmation_settings(
    cfg: DictConfig,
    calibration_results_path: Path,
    artifact_path: Path,
    source_path: Path,
) -> list[dict]:
    with calibration_results_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if not calibration.get("complete", False):
        raise ValueError("Confirmation requires a complete calibration result.")
    calibration_compatibility = calibration.get("compatibility", {})
    if calibration_compatibility.get("artifact_path") != str(artifact_path):
        raise ValueError("Calibration used a different model artifact.")
    if calibration_compatibility.get("source_sample_path") != str(source_path):
        raise ValueError("Calibration used a different unconditional source payload.")
    current_compatibility = compatibility_metadata(cfg, artifact_path, source_path)
    shared_keys = (
        "methods",
        "target_means",
        "target_variance",
        "guidance_scales",
        "guidance_starts",
        "num_steps",
        "batch_size",
        "updated_dim",
        "naive",
        "tds",
    )
    for key in shared_keys:
        if calibration_compatibility.get(key) != current_compatibility.get(key):
            raise ValueError(
                f"Confirmation setting {key!r} differs from calibration."
            )

    settings = []
    rows = calibration.get("results", [])
    for method in [str(value) for value in cfg.sweep.methods]:
        for target_mean in [float(value) for value in cfg.sweep.target_means]:
            candidates = [
                row
                for row in rows
                if row["method"] == method
                and float(row["target_mean"]) == target_mean
            ]
            if not candidates:
                raise ValueError(
                    f"No calibration candidates for method={method}, target={target_mean:g}."
                )
            winner = min(candidates, key=winner_key)
            settings.append(
                {
                    "method": method,
                    "target_mean": target_mean,
                    "guidance_scale": float(winner["guidance_scale"]),
                    "guidance_start": float(winner["guidance_start"]),
                    "selected_from": str(winner["setting_id"]),
                }
            )
    return settings


def draw_samples(
    *,
    method: str,
    model,
    sde,
    device: torch.device,
    cfg: DictConfig,
    num_samples: int,
    num_particles: int,
    target_marginal,
    analytic_original_marginal,
    model_original_marginal,
    original_mean: torch.Tensor,
    original_covariance: torch.Tensor,
    updated_mean: torch.Tensor,
    updated_covariance: torch.Tensor,
    guidance_scale: float,
    guidance_start: float,
) -> torch.Tensor:
    batches = []
    remaining = num_samples
    while remaining > 0:
        batch_n = min(int(cfg.sweep.batch_size), remaining)
        if method == "naive_guidance":
            sampler = GuidedReverseSDESampler(
                sde=sde,
                target_marginal=target_marginal,
                original_marginal=analytic_original_marginal,
                data_dim=int(cfg.dataset.dim),
                updated_dim=int(cfg.jeffrey.updated_dim),
                num_samples=batch_n,
                guidance_scale=guidance_scale,
                guidance_start=guidance_start,
            )
            batch = sampler.sample(
                model,
                int(cfg.sweep.num_steps),
                device,
                return_mean=True,
                progress=bool(cfg.sampling.progress),
            )
        else:
            tds = TDSSampler(
                num_particles=num_particles,
                sde=sde,
                num_samples=batch_n,
                target_marginal=target_marginal,
                original_marginal=model_original_marginal,
                updated_dim=int(cfg.jeffrey.updated_dim),
                data_dim=int(cfg.dataset.dim),
                num_steps=int(cfg.sweep.num_steps),
                twist_type=str(cfg.sweep.tds.twist_type),
                base_mean=original_mean,
                base_covariance=original_covariance,
                updated_mean=updated_mean,
                updated_covariance=updated_covariance,
                resample_type=str(cfg.sweep.tds.resample_type),
                guidance_scale=guidance_scale,
                guidance_ramp=str(cfg.sweep.tds.guidance_ramp),
                guidance_start=guidance_start,
                adaptive_resampling=bool(cfg.sweep.tds.adaptive_resampling),
                ess_threshold=float(cfg.sweep.tds.ess_threshold),
                max_guidance_grad_norm=cfg.sweep.tds.max_guidance_grad_norm,
            )
            batch = tds.sample(
                model,
                device,
                progress=bool(cfg.sampling.progress),
            )
        batches.append(batch.detach().cpu())
        remaining -= batch_n
    return torch.cat(batches, dim=0)


def row_from_metrics(
    *,
    setting: dict,
    metrics: dict,
    sample_path: Path,
    num_samples: int,
    num_particles: int,
    seed: int,
    cfg: DictConfig,
) -> dict:
    method = str(setting["method"])
    return {
        "setting_id": setting_id(
            method,
            setting["target_mean"],
            setting["guidance_scale"],
            setting["guidance_start"],
        ),
        "stage": str(cfg.sweep.stage),
        "method": method,
        "target_mean": float(setting["target_mean"]),
        "target_variance": float(cfg.sweep.target_variance),
        "guidance_scale": float(setting["guidance_scale"]),
        "guidance_start": float(setting["guidance_start"]),
        "guidance_schedule": (
            str(cfg.sweep.naive.schedule)
            if method == "naive_guidance"
            else str(cfg.sweep.tds.guidance_ramp)
        ),
        "num_samples": num_samples,
        "num_particles": 0 if method == "naive_guidance" else num_particles,
        "num_steps": int(cfg.sweep.num_steps),
        "seed": seed,
        "updated_marginal_mean": metrics["updated_marginal_mean"],
        "updated_marginal_std": metrics["updated_marginal_std"],
        "marginal_w2_squared": metrics["marginal_w2_squared"],
        "joint_w2_squared": metrics["joint_w2_squared"],
        "mean_error_l2": metrics["mean_error_l2"],
        "covariance_error_fro": metrics["covariance_error_fro"],
        "sample_path": str(sample_path),
        "selected_from": str(setting.get("selected_from", "")),
    }


def config_for_setting(
    cfg: DictConfig,
    setting: dict,
    num_samples: int,
    num_particles: int,
    seed: int,
) -> dict:
    saved_config = OmegaConf.to_container(cfg, resolve=True)
    saved_config["seed"] = seed
    saved_config["jeffrey"]["target"]["mean"] = float(setting["target_mean"])
    saved_config["jeffrey"]["target"]["variance"] = float(
        cfg.sweep.target_variance
    )
    saved_config["sampling"]["num_samples"] = num_samples
    saved_config["sampling"]["batch_size"] = int(cfg.sweep.batch_size)
    saved_config["sampling"]["num_steps"] = int(cfg.sweep.num_steps)
    if setting["method"] == "naive_guidance":
        saved_config["naive_guidance"]["guidance_scale"] = float(
            setting["guidance_scale"]
        )
        saved_config["naive_guidance"]["guidance_coeff"] = None
        saved_config["naive_guidance"]["guidance_start"] = float(
            setting["guidance_start"]
        )
    else:
        saved_config["sampler"].update(
            {
                "num_particles": num_particles,
                "twist_type": str(cfg.sweep.tds.twist_type),
                "guidance_scale": float(setting["guidance_scale"]),
                "guidance_ramp": str(cfg.sweep.tds.guidance_ramp),
                "guidance_start": float(setting["guidance_start"]),
                "resample_type": str(cfg.sweep.tds.resample_type),
                "adaptive_resampling": bool(cfg.sweep.tds.adaptive_resampling),
                "ess_threshold": float(cfg.sweep.tds.ess_threshold),
                "max_guidance_grad_norm": cfg.sweep.tds.max_guidance_grad_norm,
            }
        )
    return saved_config


def save_progress(
    *,
    sweep_dir: Path,
    compatibility: dict,
    rows: list[dict],
    reference_rows: list[dict],
    expected_count: int,
) -> None:
    complete = len(rows) == expected_count
    document = {
        "complete": complete,
        "num_completed": len(rows),
        "num_expected": expected_count,
        "compatibility": compatibility,
        "reference_results": reference_rows,
        "results": rows,
    }
    atomic_json_save(document, sweep_dir / "results.json")
    atomic_csv_save(rows, sweep_dir / "results.csv")


def save_calibration_plot(
    rows: list[dict],
    reference_rows: list[dict],
    output_path: Path,
) -> None:
    methods = [
        method
        for method in ("naive_guidance", "tds")
        if any(row["method"] == method for row in rows)
    ]
    targets = sorted({float(row["target_mean"]) for row in rows})
    fig, axes = plt.subplots(
        len(methods),
        len(targets),
        figsize=(6 * len(targets), 4.8 * len(methods)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    all_values = [float(row["marginal_w2_squared"]) for row in rows]
    value_min = min(all_values)
    value_max = max(all_values)
    for row_index, method in enumerate(methods):
        for column_index, target_mean in enumerate(targets):
            ax = axes[row_index, column_index]
            subset = [
                row
                for row in rows
                if row["method"] == method
                and float(row["target_mean"]) == target_mean
            ]
            scales = sorted({float(row["guidance_scale"]) for row in subset})
            starts = sorted({float(row["guidance_start"]) for row in subset})
            values = torch.full((len(starts), len(scales)), torch.nan)
            for row in subset:
                i = starts.index(float(row["guidance_start"]))
                j = scales.index(float(row["guidance_scale"]))
                values[i, j] = float(row["marginal_w2_squared"])
            image = ax.imshow(
                values.numpy(),
                aspect="auto",
                cmap="viridis_r",
                vmin=value_min,
                vmax=value_max,
            )
            for i in range(len(starts)):
                for j in range(len(scales)):
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.3g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            reference = next(
                row
                for row in reference_rows
                if float(row["target_mean"]) == target_mean
            )
            ax.set_title(
                f"{method}, target mean={target_mean:g}\n"
                f"source W2²={reference['marginal_w2_squared']:.3g}"
            )
            ax.set_xticks(range(len(scales)), [f"{value:g}" for value in scales])
            ax.set_yticks(range(len(starts)), [f"{value:g}" for value in starts])
            ax.set_xlabel("guidance scale")
            ax.set_ylabel("guidance start")
    if image is not None:
        fig.colorbar(image, ax=axes, label="updated-marginal W2²", shrink=0.8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confirmation_plot(rows: list[dict], output_path: Path) -> None:
    methods = [
        method
        for method in ("naive_guidance", "tds")
        if any(row["method"] == method for row in rows)
    ]
    targets = sorted({float(row["target_mean"]) for row in rows})
    fig, axes = plt.subplots(
        len(methods),
        len(targets),
        figsize=(6 * len(targets), 5 * len(methods)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, method in enumerate(methods):
        for column_index, target_mean in enumerate(targets):
            row = next(
                value
                for value in rows
                if value["method"] == method
                and float(value["target_mean"]) == target_mean
            )
            payload = torch.load(row["sample_path"], map_location="cpu")
            samples = payload["samples"]
            ax = axes[row_index, column_index]
            preview = samples[: min(3000, samples.shape[0])]
            ax.scatter(
                preview[:, 0].numpy(),
                preview[:, 1].numpy(),
                s=5,
                alpha=0.2,
                linewidths=0,
                color="#2563eb",
            )
            plot_gaussian_contours(
                ax,
                torch.as_tensor(payload["updated_mean"], dtype=torch.float32),
                torch.as_tensor(payload["updated_covariance"], dtype=torch.float32),
                color="#7c3aed",
                linestyle="-.",
                label="exact Jeffrey",
            )
            plot_gaussian_contours(
                ax,
                torch.as_tensor(payload["sample_mean"], dtype=torch.float32),
                torch.as_tensor(payload["sample_covariance"], dtype=torch.float32),
                color="#f97316",
                linestyle="--",
                label="sample fit",
            )
            ax.set_title(
                f"{method}, target={target_mean:g}\n"
                f"scale={row['guidance_scale']:g}, start={row['guidance_start']:g}, "
                f"W2²={row['marginal_w2_squared']:.3g}"
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    if cfg.sampling.artifact_path is None:
        raise ValueError("An explicit sampling.artifact_path is required for the sweep.")
    if cfg.jeffrey.source_sample_path is None:
        raise ValueError("An explicit jeffrey.source_sample_path is required for the sweep.")

    artifact_path = resolve_path(project_root, str(cfg.sampling.artifact_path))
    source_path = resolve_path(project_root, str(cfg.jeffrey.source_sample_path))
    source_samples, source_payload = load_source_samples(source_path, artifact_path)
    artifact_payload = torch.load(artifact_path, map_location=device)

    model = instantiate(cfg.model).to(device)
    model.load_state_dict(load_model_state(artifact_payload))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    sde = instantiate(cfg.sde)

    updated_dim = int(cfg.jeffrey.updated_dim)
    analytic_original_marginal = torch.distributions.Normal(
        float(cfg.dataset.mean[updated_dim]),
        float(cfg.dataset.covariance[updated_dim][updated_dim]) ** 0.5,
    )
    model_original_marginal = estimate_model_marginal(source_samples, updated_dim)
    original_mean, original_covariance = estimate_model_gaussian_params(source_samples)

    sweep_dir = make_sweep_dir(project_root, cfg)
    compatibility = compatibility_metadata(cfg, artifact_path, source_path)
    results_path = sweep_dir / "results.json"
    rows: list[dict] = []
    if bool(cfg.sweep.resume):
        if not results_path.is_file():
            raise FileNotFoundError(f"No resumable results at {results_path}.")
        with results_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if prior.get("compatibility") != compatibility:
            raise ValueError("Resume settings do not match the saved sweep.")
        rows = prior.get("results", [])
        missing_samples = [
            row["sample_path"]
            for row in rows
            if not Path(str(row["sample_path"])).is_file()
        ]
        if missing_samples:
            raise FileNotFoundError(
                "Resume state references missing sample payloads: "
                + ", ".join(missing_samples)
            )

    if str(cfg.sweep.stage) == "calibration":
        settings = make_calibration_settings(cfg)
    else:
        calibration_path = resolve_path(
            project_root,
            str(cfg.sweep.calibration_results_path),
        )
        settings = make_confirmation_settings(
            cfg,
            calibration_path,
            artifact_path,
            source_path,
        )

    reference_rows = []
    exact_params = {}
    for target_mean in [float(value) for value in cfg.sweep.target_means]:
        target_marginal = GaussianTargetMarginal(
            target_mean,
            float(cfg.sweep.target_variance),
        )
        updated_mean, updated_covariance = jeffrey_updated_gaussian_params(
            cfg.dataset.mean,
            cfg.dataset.covariance,
            updated_dim,
            target_marginal,
        )
        exact_params[target_mean] = (
            target_marginal,
            updated_mean,
            updated_covariance,
        )
        metrics = sample_metrics(
            source_samples,
            updated_dim,
            updated_mean,
            updated_covariance,
        )
        reference_rows.append(
            {
                "target_mean": target_mean,
                "target_variance": float(cfg.sweep.target_variance),
                "updated_marginal_mean": metrics["updated_marginal_mean"],
                "updated_marginal_std": metrics["updated_marginal_std"],
                "marginal_w2_squared": metrics["marginal_w2_squared"],
                "joint_w2_squared": metrics["joint_w2_squared"],
                "mean_error_l2": metrics["mean_error_l2"],
                "covariance_error_fro": metrics["covariance_error_fro"],
            }
        )

    completed_ids = {str(row["setting_id"]) for row in rows}
    num_samples, num_particles, seed = stage_budget(cfg)
    for index, setting in enumerate(settings, start=1):
        identifier = setting_id(
            setting["method"],
            setting["target_mean"],
            setting["guidance_scale"],
            setting["guidance_start"],
        )
        if identifier in completed_ids:
            print(f"Skipping completed setting: {identifier}")
            continue

        print(f"[{index}/{len(settings)}] Running {identifier}")
        torch.manual_seed(seed)
        target_marginal, updated_mean, updated_covariance = exact_params[
            float(setting["target_mean"])
        ]
        samples = draw_samples(
            method=str(setting["method"]),
            model=model,
            sde=sde,
            device=device,
            cfg=cfg,
            num_samples=num_samples,
            num_particles=num_particles,
            target_marginal=target_marginal,
            analytic_original_marginal=analytic_original_marginal,
            model_original_marginal=model_original_marginal,
            original_mean=original_mean,
            original_covariance=original_covariance,
            updated_mean=updated_mean,
            updated_covariance=updated_covariance,
            guidance_scale=float(setting["guidance_scale"]),
            guidance_start=float(setting["guidance_start"]),
        )
        metrics = sample_metrics(
            samples,
            updated_dim,
            updated_mean,
            updated_covariance,
        )

        if setting["method"] == "tds":
            run_label = make_tds_run_label(
                str(cfg.sweep.tds.twist_type),
                float(setting["guidance_scale"]),
                str(cfg.sweep.tds.guidance_ramp),
                float(setting["guidance_start"]),
                str(cfg.sweep.tds.resample_type),
                bool(cfg.sweep.tds.adaptive_resampling),
                float(cfg.sweep.tds.ess_threshold),
                num_particles,
                int(cfg.sweep.num_steps),
            )
        else:
            run_label = (
                f"scale{float(setting['guidance_scale']):g}_"
                f"guidance_start_{float(setting['guidance_start']):g}_"
                f"T{int(cfg.sweep.num_steps)}"
            )

        sample_path = sweep_dir / "samples" / f"{identifier}.pt"
        payload = {
            "samples": samples,
            "sample_type": str(setting["method"]),
            "run_label": run_label,
            "guidance_scale": float(setting["guidance_scale"]),
            "guidance_start": float(setting["guidance_start"]),
            "num_steps": int(cfg.sweep.num_steps),
            "seed": seed,
            "artifact_path": str(artifact_path),
            "source_sample_path": str(source_path),
            "source_artifact_path": str(source_payload["artifact_path"]),
            "config": config_for_setting(
                cfg,
                setting,
                num_samples,
                num_particles,
                seed,
            ),
            "sweep_stage": str(cfg.sweep.stage),
            "sweep_setting_id": identifier,
            "selected_from": str(setting.get("selected_from", "")),
            "complete": True,
            "num_completed": num_samples,
            "updated_dim": updated_dim,
            "target_marginal": {
                "type": "gaussian",
                "mean": target_marginal.mean,
                "variance": target_marginal.variance,
            },
            "updated_mean": updated_mean,
            "updated_covariance": updated_covariance,
            "sample_mean": metrics["sample_mean"],
            "sample_covariance": metrics["sample_covariance"],
        }
        if setting["method"] == "naive_guidance":
            payload.update(
                {
                    "guidance_coeff": float(setting["guidance_scale"]),
                    "guidance_schedule": str(cfg.sweep.naive.schedule),
                    "original_marginal": {
                        "type": "gaussian",
                        "source": "analytic_dataset",
                        "mean": float(analytic_original_marginal.mean),
                        "variance": float(analytic_original_marginal.variance),
                    },
                }
            )
        else:
            payload.update(
                {
                    "twist_type": str(cfg.sweep.tds.twist_type),
                    "guidance_ramp": str(cfg.sweep.tds.guidance_ramp),
                    "resample_type": str(cfg.sweep.tds.resample_type),
                    "adaptive_resampling": bool(cfg.sweep.tds.adaptive_resampling),
                    "ess_threshold": float(cfg.sweep.tds.ess_threshold),
                    "max_guidance_grad_norm": cfg.sweep.tds.max_guidance_grad_norm,
                    "num_particles": num_particles,
                    "original_marginal": {
                        "type": "gaussian",
                        "source": "model_samples",
                        "mean": float(model_original_marginal.mean),
                        "variance": float(model_original_marginal.variance),
                    },
                }
            )
        atomic_torch_save(payload, sample_path)

        row = row_from_metrics(
            setting=setting,
            metrics=metrics,
            sample_path=sample_path,
            num_samples=num_samples,
            num_particles=num_particles,
            seed=seed,
            cfg=cfg,
        )
        rows.append(row)
        completed_ids.add(identifier)
        save_progress(
            sweep_dir=sweep_dir,
            compatibility=compatibility,
            rows=rows,
            reference_rows=reference_rows,
            expected_count=len(settings),
        )
        print(
            f"  marginal W2²={row['marginal_w2_squared']:.6g}, "
            f"joint W2²={row['joint_w2_squared']:.6g}"
        )

    if len(rows) != len(settings):
        raise RuntimeError("Sweep ended before all expected settings were recorded.")
    if str(cfg.sweep.stage) == "calibration":
        save_calibration_plot(
            rows,
            reference_rows,
            sweep_dir / "marginal_w2_heatmaps.png",
        )
    else:
        save_confirmation_plot(rows, sweep_dir / "confirmation_comparison.png")
    print(f"Saved complete {cfg.sweep.stage} sweep to: {sweep_dir}")


if __name__ == "__main__":
    main()
