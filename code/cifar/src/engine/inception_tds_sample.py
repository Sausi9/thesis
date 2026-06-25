from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid, save_image

from src.jeffrey.inception_ratio import InceptionRatioPotential, compute_ratio_logits
from src.samplers.inception_tds import InceptionTDSSampler
from src.utils import (
    find_latest_artifact,
    load_model_state,
    resolve_device,
    resolve_path,
    timestamped_output_path,
)


def make_run_label(
    guidance_scale: float,
    guidance_ramp: str | None,
    guidance_start: float,
    resample_type: str,
    adaptive_resampling: bool,
    ess_threshold: float,
    num_particles: int,
    num_steps: int,
) -> str:
    mode = "adaptive" if adaptive_resampling else "always"
    threshold = f"_ess{ess_threshold:g}" if adaptive_resampling else ""
    ramp = "" if guidance_ramp is None else f"_{guidance_ramp}-ramp"
    return (
        f"inception_ratio_scale{guidance_scale:g}{ramp}_"
        f"guidance_start_{guidance_start}_"
        f"{resample_type}_{mode}{threshold}_K{num_particles}_T{num_steps}"
    )


def make_output_path(cfg: DictConfig, project_root: Path, run_name: str, run_label: str) -> Path:
    return timestamped_output_path(
        output_dir=resolve_path(project_root, str(cfg.sampling.output_dir)),
        output_name=cfg.sampling.output_name,
        default_stem=f"{run_name}_inception_tds_samples_{run_label}",
        extension=".pt",
    )


def make_preview_path(cfg: DictConfig, project_root: Path, output_path: Path) -> Path:
    preview_dir = resolve_path(project_root, str(cfg.sampling.preview_dir))
    preview_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir / f"{output_path.stem}.png"


def optional_float(value):
    if value is None:
        return None
    return float(value)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))

    if str(cfg.jeffrey.feature) != "inception_ratio":
        raise ValueError(
            "inception_tds_sample requires jeffrey=inception_ratio, "
            f"got feature={cfg.jeffrey.feature!r}."
        )

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
    ratio_classifier_path = resolve_path(project_root, str(cfg.jeffrey.ratio_classifier_path))
    ratio_potential = InceptionRatioPotential(
        classifier_path=ratio_classifier_path,
        feature_extractor=str(cfg.jeffrey.feature_extractor),
        feature_layer=str(cfg.jeffrey.feature_layer),
        preprocessing=str(cfg.jeffrey.preprocessing),
        logit_clip=optional_float(cfg.jeffrey.logit_clip),
    ).to(device)
    ratio_potential.eval()

    guidance_scale = float(cfg.jeffrey.guidance_scale)
    guidance_ramp = cfg.sampler.guidance_ramp
    guidance_start = float(cfg.sampler.guidance_start)
    resample_type = str(cfg.sampler.resample_type)
    adaptive_resampling = bool(cfg.sampler.adaptive_resampling)
    ess_threshold = float(cfg.sampler.ess_threshold)
    max_guidance_grad_norm = cfg.sampler.max_guidance_grad_norm
    num_particles = int(cfg.sampler.num_particles)
    num_steps = int(cfg.sampling.num_steps)
    sample_shape = tuple(int(v) for v in cfg.dataset.shape)
    run_label = make_run_label(
        guidance_scale,
        guidance_ramp,
        guidance_start,
        resample_type,
        adaptive_resampling,
        ess_threshold,
        num_particles,
        num_steps,
    )

    all_samples = []
    remaining_samples = int(cfg.sampling.num_samples)
    while remaining_samples > 0:
        batch_n = min(int(cfg.sampling.batch_size), remaining_samples)
        tds = InceptionTDSSampler(
            num_particles=num_particles,
            sde=sde,
            num_samples=batch_n,
            sample_shape=sample_shape,
            ratio_potential=ratio_potential,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            resample_type=resample_type,
            guidance_ramp=guidance_ramp,
            guidance_start=guidance_start,
            adaptive_resampling=adaptive_resampling,
            ess_threshold=ess_threshold,
            max_guidance_grad_norm=max_guidance_grad_norm,
        )
        samples_batch = tds.sample(model, device, progress=bool(cfg.sampling.progress))
        all_samples.append(samples_batch.detach().cpu())
        remaining_samples -= batch_n

    samples = torch.cat(all_samples, dim=0)
    ratio_logits = compute_ratio_logits(
        potential=ratio_potential,
        samples=samples,
        device=device,
        batch_size=int(cfg.ratio.batch_size),
    )

    run_name = str(payload.get("run_name") or artifact_path.stem)
    output_path = make_output_path(cfg, project_root, run_name, run_label)
    result = {
        "samples": samples.detach().cpu(),
        "sample_type": "inception_tds",
        "run_label": run_label,
        "feature": "inception_ratio",
        "ratio_classifier_path": str(ratio_classifier_path),
        "guidance_scale": guidance_scale,
        "logit_clip": optional_float(cfg.jeffrey.logit_clip),
        "preprocessing": str(cfg.jeffrey.preprocessing),
        "feature_extractor": str(cfg.jeffrey.feature_extractor),
        "feature_layer": str(cfg.jeffrey.feature_layer),
        "guidance_ramp": guidance_ramp,
        "guidance_start": guidance_start,
        "resample_type": resample_type,
        "adaptive_resampling": adaptive_resampling,
        "ess_threshold": ess_threshold,
        "max_guidance_grad_norm": max_guidance_grad_norm,
        "num_particles": num_particles,
        "num_steps": num_steps,
        "seed": int(cfg.seed),
        "artifact_path": str(artifact_path),
        "requested_weight_type": requested_weight_type,
        "loaded_weight_type": loaded_weight_type,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "image_shape": sample_shape,
        "ratio_logit_mean": ratio_logits.mean().detach().cpu(),
        "ratio_logit_std": ratio_logits.std(
            unbiased=ratio_logits.numel() > 1
        ).detach().cpu(),
    }
    torch.save(result, output_path)

    if bool(cfg.sampling.save_preview):
        preview_count = min(int(cfg.sampling.preview_num_samples), samples.shape[0])
        preview = (samples[:preview_count].clamp(-1, 1) + 1) / 2
        grid = make_grid(preview, nrow=int(cfg.sampling.preview_nrow))
        preview_path = make_preview_path(cfg, project_root, output_path)
        save_image(grid, preview_path)
        print(f"Saved preview to: {preview_path}")

    print(f"Loaded artifact: {artifact_path}")
    print(f"Loaded weight type: {loaded_weight_type} (requested: {requested_weight_type})")
    print(f"Loaded ratio classifier: {ratio_classifier_path}")
    print(f"Guidance scale: {guidance_scale:g}")
    print(f"Saved Inception TDS samples to: {output_path}")
    print(f"Ratio logit mean: {float(result['ratio_logit_mean']):.6f}")
    print(f"Ratio logit std: {float(result['ratio_logit_std']):.6f}")


if __name__ == "__main__":
    main()
