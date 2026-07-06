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


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_existing_samples(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if "samples" not in payload:
        raise KeyError(f"Resume file {path} does not contain 'samples'.")
    return payload


def ensure_resume_compatible(existing: dict, expected: dict, path: Path) -> None:
    existing_compat = existing.get("resume_compatibility")
    if not isinstance(existing_compat, dict):
        raise ValueError(
            f"Cannot resume from {path}: missing resume_compatibility metadata."
        )

    mismatches = []
    for key, expected_value in expected.items():
        existing_value = existing_compat.get(key)
        if existing_value != expected_value:
            mismatches.append((key, existing_value, expected_value))

    if mismatches:
        formatted = "\n".join(
            f"  {key}: existing={old!r}, requested={new!r}"
            for key, old, new in mismatches
        )
        raise ValueError(f"Cannot resume from {path}; incompatible settings:\n{formatted}")


def make_result_payload(
    *,
    samples: torch.Tensor,
    complete: bool,
    num_completed: int,
    num_batches_completed: int,
    run_label: str,
    ratio_classifier_path: Path,
    guidance_scale: float,
    guidance_ramp,
    guidance_start: float,
    resample_type: str,
    adaptive_resampling: bool,
    ess_threshold: float,
    max_guidance_grad_norm,
    num_particles: int,
    num_steps: int,
    seed: int,
    artifact_path: Path,
    requested_weight_type: str,
    loaded_weight_type: str,
    sample_shape: tuple[int, ...],
    cfg: DictConfig,
    resume_compatibility: dict,
    ratio_logits: torch.Tensor | None = None,
) -> dict:
    payload = {
        "samples": samples.detach().cpu(),
        "sample_type": "inception_tds",
        "complete": bool(complete),
        "num_completed": int(num_completed),
        "num_batches_completed": int(num_batches_completed),
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
        "seed": int(seed),
        "artifact_path": str(artifact_path),
        "requested_weight_type": requested_weight_type,
        "loaded_weight_type": loaded_weight_type,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "image_shape": sample_shape,
        "resume_compatibility": resume_compatibility,
    }

    if ratio_logits is not None:
        payload["ratio_logit_mean"] = ratio_logits.mean().detach().cpu()
        payload["ratio_logit_std"] = ratio_logits.std(
            unbiased=ratio_logits.numel() > 1
        ).detach().cpu()

    return payload


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    base_seed = int(cfg.seed)
    torch.manual_seed(base_seed)

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
    run_name = str(payload.get("run_name") or artifact_path.stem)
    output_path = make_output_path(cfg, project_root, run_name, run_label)

    resume_enabled = bool(OmegaConf.select(cfg, "sampling.resume", default=False))
    if resume_enabled and cfg.sampling.output_name is None:
        raise ValueError("sampling.resume=true requires a fixed sampling.output_name.")

    save_every_batches = int(OmegaConf.select(cfg, "sampling.save_every_batches", default=1))
    if save_every_batches <= 0:
        raise ValueError("sampling.save_every_batches must be positive.")

    target_num_samples = int(cfg.sampling.num_samples)
    batch_size = int(cfg.sampling.batch_size)
    resume_compatibility = {
        "artifact_path": str(artifact_path),
        "ratio_classifier_path": str(ratio_classifier_path),
        "guidance_scale": guidance_scale,
        "guidance_ramp": None if guidance_ramp is None else str(guidance_ramp),
        "guidance_start": guidance_start,
        "resample_type": resample_type,
        "adaptive_resampling": adaptive_resampling,
        "ess_threshold": ess_threshold,
        "max_guidance_grad_norm": None
        if max_guidance_grad_norm is None
        else float(max_guidance_grad_norm),
        "num_particles": num_particles,
        "num_steps": num_steps,
        "seed": base_seed,
        "sample_shape": list(sample_shape),
        "batch_size": batch_size,
        "requested_weight_type": requested_weight_type,
        "loaded_weight_type": loaded_weight_type,
        "feature_extractor": str(cfg.jeffrey.feature_extractor),
        "feature_layer": str(cfg.jeffrey.feature_layer),
        "preprocessing": str(cfg.jeffrey.preprocessing),
        "logit_clip": optional_float(cfg.jeffrey.logit_clip),
    }

    existing_payload = load_existing_samples(output_path) if resume_enabled else None
    all_samples = []
    num_completed = 0
    num_batches_completed = 0
    if existing_payload is not None:
        ensure_resume_compatible(existing_payload, resume_compatibility, output_path)
        existing_samples = existing_payload["samples"].detach().cpu()
        if existing_samples.ndim != 4:
            raise ValueError(
                f"Expected existing samples [N, C, H, W], got {tuple(existing_samples.shape)}."
            )
        if tuple(existing_samples.shape[1:]) != sample_shape:
            raise ValueError(
                "Existing sample shape mismatch: "
                f"existing={tuple(existing_samples.shape[1:])}, requested={sample_shape}."
            )
        if existing_samples.shape[0] > target_num_samples:
            raise ValueError(
                f"Resume file already has {existing_samples.shape[0]} samples, "
                f"more than requested target {target_num_samples}."
            )
        all_samples.append(existing_samples)
        num_completed = int(existing_payload.get("num_completed", existing_samples.shape[0]))
        num_batches_completed = int(
            existing_payload.get(
                "num_batches_completed",
                (num_completed + batch_size - 1) // batch_size,
            )
        )
        print(
            f"Resuming {output_path}: "
            f"{num_completed}/{target_num_samples} samples already saved."
        )

    remaining_samples = target_num_samples - num_completed
    while remaining_samples > 0:
        batch_n = min(batch_size, remaining_samples)
        torch.manual_seed(base_seed + num_batches_completed)
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
        num_completed += batch_n
        num_batches_completed += 1

        if resume_enabled and (
            num_batches_completed % save_every_batches == 0 or remaining_samples == 0
        ):
            partial_samples = torch.cat(all_samples, dim=0)
            partial_payload = make_result_payload(
                samples=partial_samples,
                complete=False,
                num_completed=num_completed,
                num_batches_completed=num_batches_completed,
                run_label=run_label,
                ratio_classifier_path=ratio_classifier_path,
                guidance_scale=guidance_scale,
                guidance_ramp=guidance_ramp,
                guidance_start=guidance_start,
                resample_type=resample_type,
                adaptive_resampling=adaptive_resampling,
                ess_threshold=ess_threshold,
                max_guidance_grad_norm=max_guidance_grad_norm,
                num_particles=num_particles,
                num_steps=num_steps,
                seed=base_seed,
                artifact_path=artifact_path,
                requested_weight_type=requested_weight_type,
                loaded_weight_type=loaded_weight_type,
                sample_shape=sample_shape,
                cfg=cfg,
                resume_compatibility=resume_compatibility,
            )
            atomic_torch_save(partial_payload, output_path)
            print(f"Saved resumable shard progress: {num_completed}/{target_num_samples}")

    samples = torch.cat(all_samples, dim=0)
    ratio_logits = compute_ratio_logits(
        potential=ratio_potential,
        samples=samples,
        device=device,
        batch_size=int(cfg.ratio.batch_size),
    )

    result = make_result_payload(
        samples=samples,
        complete=True,
        num_completed=int(samples.shape[0]),
        num_batches_completed=num_batches_completed,
        run_label=run_label,
        ratio_classifier_path=ratio_classifier_path,
        guidance_scale=guidance_scale,
        guidance_ramp=guidance_ramp,
        guidance_start=guidance_start,
        resample_type=resample_type,
        adaptive_resampling=adaptive_resampling,
        ess_threshold=ess_threshold,
        max_guidance_grad_norm=max_guidance_grad_norm,
        num_particles=num_particles,
        num_steps=num_steps,
        seed=base_seed,
        artifact_path=artifact_path,
        requested_weight_type=requested_weight_type,
        loaded_weight_type=loaded_weight_type,
        sample_shape=sample_shape,
        cfg=cfg,
        resume_compatibility=resume_compatibility,
        ratio_logits=ratio_logits,
    )
    atomic_torch_save(result, output_path)

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
