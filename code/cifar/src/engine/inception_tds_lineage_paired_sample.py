from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.engine.inception_tds_paired_sample import pixel_distances, save_analysis
from src.engine.inception_tds_sample import (
    atomic_torch_save,
    ensure_resume_compatible,
    make_run_label,
    optional_float,
)
from src.jeffrey.inception_ratio import InceptionRatioPotential, compute_ratio_logits
from src.samplers.lineage_replay_inception_tds import (
    LineageReplayInceptionTDSSampler,
)
from src.utils import (
    find_latest_artifact,
    load_model_state,
    resolve_device,
    resolve_path,
    timestamped_output_path,
)


PAIRING_SEMANTICS = "selected_lineage_unguided_replay"
PAIRING_CAVEAT = (
    "Unguided images follow ordinary VP-SDE transitions with no particles, "
    "weights, or resampling. Their starting lineage and replayed Gaussian "
    "increments are selected by guided TDS, so they are counterfactual "
    "counterparts rather than independent unconditional samples."
)
RNG_SCHEME = "four_stream_batch_seed_v1"


def make_output_path(
    cfg: DictConfig,
    project_root: Path,
    run_name: str,
    run_label: str,
) -> Path:
    return timestamped_output_path(
        output_dir=resolve_path(project_root, str(cfg.sampling.output_dir)),
        output_name=cfg.sampling.output_name,
        default_stem=f"{run_name}_lineage_paired_inception_tds_{run_label}",
        extension=".pt",
    )


def make_payload(
    *,
    guided_samples: torch.Tensor,
    unguided_samples: torch.Tensor,
    selected_particle_indices: torch.Tensor,
    selected_boundary_indices: torch.Tensor,
    batch_resampling_steps: list[int],
    batch_replay_start_steps: list[int],
    batch_replay_num_steps: list[int],
    complete: bool,
    num_completed: int,
    num_batches_completed: int,
    run_label: str,
    ratio_classifier_path: Path,
    artifact_path: Path,
    requested_weight_type: str,
    loaded_weight_type: str,
    cfg: DictConfig,
    resume_compatibility: dict,
    pixel_l2: torch.Tensor | None = None,
    pixel_mse: torch.Tensor | None = None,
    guided_ratio_logits: torch.Tensor | None = None,
    unguided_ratio_logits: torch.Tensor | None = None,
) -> dict:
    guided_cpu = guided_samples.detach().cpu()
    payload = {
        "samples": guided_cpu,
        "guided_samples": guided_cpu,
        "unguided_samples": unguided_samples.detach().cpu(),
        "selected_particle_indices": selected_particle_indices.detach().cpu().long(),
        "selected_boundary_indices": selected_boundary_indices.detach().cpu().long(),
        "batch_resampling_steps": [int(value) for value in batch_resampling_steps],
        "batch_replay_start_steps": [
            int(value) for value in batch_replay_start_steps
        ],
        "batch_replay_num_steps": [int(value) for value in batch_replay_num_steps],
        "sample_type": "inception_tds_lineage_paired",
        "pairing_semantics": PAIRING_SEMANTICS,
        "pairing_caveat": PAIRING_CAVEAT,
        "unguided_uses_particles": False,
        "unguided_uses_weights": False,
        "unguided_uses_resampling": False,
        "unguided_counterparts_are_independent": False,
        "rng_scheme": RNG_SCHEME,
        "complete": bool(complete),
        "num_completed": int(num_completed),
        "num_batches_completed": int(num_batches_completed),
        "run_label": run_label,
        "feature": "inception_ratio",
        "ratio_classifier_path": str(ratio_classifier_path),
        "guidance_scale": float(cfg.jeffrey.guidance_scale),
        "guidance_ramp": cfg.sampler.guidance_ramp,
        "guidance_start": float(cfg.sampler.guidance_start),
        "logit_clip": optional_float(cfg.jeffrey.logit_clip),
        "preprocessing": str(cfg.jeffrey.preprocessing),
        "feature_extractor": str(cfg.jeffrey.feature_extractor),
        "feature_layer": str(cfg.jeffrey.feature_layer),
        "resample_type": str(cfg.sampler.resample_type),
        "adaptive_resampling": bool(cfg.sampler.adaptive_resampling),
        "ess_threshold": float(cfg.sampler.ess_threshold),
        "max_guidance_grad_norm": optional_float(
            cfg.sampler.max_guidance_grad_norm
        ),
        "num_particles": int(cfg.sampler.num_particles),
        "num_steps": int(cfg.sampling.num_steps),
        "seed": int(cfg.seed),
        "artifact_path": str(artifact_path),
        "requested_weight_type": requested_weight_type,
        "loaded_weight_type": loaded_weight_type,
        "image_shape": tuple(int(value) for value in cfg.dataset.shape),
        "resume_compatibility": resume_compatibility,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if pixel_l2 is not None:
        payload["pixel_l2"] = pixel_l2.detach().cpu()
        payload["pixel_mse"] = pixel_mse.detach().cpu()
        payload["guided_ratio_logits"] = guided_ratio_logits.detach().cpu()
        payload["unguided_ratio_logits"] = unguided_ratio_logits.detach().cpu()
    return payload


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    base_seed = int(cfg.seed)
    torch.manual_seed(base_seed)

    if str(cfg.jeffrey.feature) != "inception_ratio":
        raise ValueError("Lineage pairing requires jeffrey=inception_ratio.")

    artifact_dir = project_root / str(cfg.training.artifacts_dir)
    artifact_path = (
        find_latest_artifact(
            artifact_dir,
            preference=str(cfg.sampling.artifact_preference),
        )
        if cfg.sampling.artifact_path is None
        else resolve_path(project_root, str(cfg.sampling.artifact_path))
    )
    artifact = torch.load(artifact_path, map_location=device)
    model = instantiate(cfg.model).to(device)
    requested_weight_type = str(cfg.sampling.weight_type)
    model_state, loaded_weight_type = load_model_state(
        artifact,
        weight_type=requested_weight_type,
        return_weight_type=True,
    )
    model.load_state_dict(model_state)
    model.eval()
    model.requires_grad_(False)

    sde = instantiate(cfg.sde)
    ratio_classifier_path = resolve_path(
        project_root, str(cfg.jeffrey.ratio_classifier_path)
    )
    ratio_potential = InceptionRatioPotential(
        classifier_path=ratio_classifier_path,
        feature_extractor=str(cfg.jeffrey.feature_extractor),
        feature_layer=str(cfg.jeffrey.feature_layer),
        preprocessing=str(cfg.jeffrey.preprocessing),
        logit_clip=optional_float(cfg.jeffrey.logit_clip),
    ).to(device)
    ratio_potential.eval()

    sample_shape = tuple(int(value) for value in cfg.dataset.shape)
    target_num_samples = int(cfg.sampling.num_samples)
    batch_size = int(cfg.sampling.batch_size)
    top_k = int(cfg.paired.top_k)
    preview_pairs = int(cfg.paired.preview_pairs)
    if target_num_samples <= 0 or batch_size <= 0:
        raise ValueError("sampling.num_samples and sampling.batch_size must be positive.")
    if top_k <= 0 or top_k > target_num_samples:
        raise ValueError("paired.top_k must be between 1 and sampling.num_samples.")
    if preview_pairs <= 0:
        raise ValueError("paired.preview_pairs must be positive.")

    run_label = make_run_label(
        float(cfg.jeffrey.guidance_scale),
        cfg.sampler.guidance_ramp,
        float(cfg.sampler.guidance_start),
        str(cfg.sampler.resample_type),
        bool(cfg.sampler.adaptive_resampling),
        float(cfg.sampler.ess_threshold),
        int(cfg.sampler.num_particles),
        int(cfg.sampling.num_steps),
    )
    run_name = str(artifact.get("run_name") or artifact_path.stem)
    output_path = make_output_path(cfg, project_root, run_name, run_label)
    analysis_dir = (
        resolve_path(project_root, str(cfg.paired.output_dir)) / output_path.stem
    )

    resume_enabled = bool(cfg.sampling.resume)
    if resume_enabled and cfg.sampling.output_name is None:
        raise ValueError("sampling.resume=true requires sampling.output_name.")
    save_every_batches = int(cfg.sampling.save_every_batches)
    if save_every_batches <= 0:
        raise ValueError("sampling.save_every_batches must be positive.")

    resume_compatibility = {
        "pairing_semantics": PAIRING_SEMANTICS,
        "rng_scheme": RNG_SCHEME,
        "artifact_path": str(artifact_path),
        "ratio_classifier_path": str(ratio_classifier_path),
        "guidance_scale": float(cfg.jeffrey.guidance_scale),
        "guidance_ramp": None
        if cfg.sampler.guidance_ramp is None
        else str(cfg.sampler.guidance_ramp),
        "guidance_start": float(cfg.sampler.guidance_start),
        "resample_type": str(cfg.sampler.resample_type),
        "adaptive_resampling": bool(cfg.sampler.adaptive_resampling),
        "ess_threshold": float(cfg.sampler.ess_threshold),
        "max_guidance_grad_norm": optional_float(
            cfg.sampler.max_guidance_grad_norm
        ),
        "num_particles": int(cfg.sampler.num_particles),
        "num_steps": int(cfg.sampling.num_steps),
        "sample_shape": list(sample_shape),
        "batch_size": batch_size,
        "seed": base_seed,
        "requested_weight_type": requested_weight_type,
        "loaded_weight_type": loaded_weight_type,
        "feature_extractor": str(cfg.jeffrey.feature_extractor),
        "feature_layer": str(cfg.jeffrey.feature_layer),
        "preprocessing": str(cfg.jeffrey.preprocessing),
        "logit_clip": optional_float(cfg.jeffrey.logit_clip),
    }

    all_guided = []
    all_unguided = []
    all_selected = []
    all_boundary = []
    batch_resampling_steps = []
    batch_replay_start_steps = []
    batch_replay_num_steps = []
    num_completed = 0
    num_batches_completed = 0

    if resume_enabled and output_path.exists():
        existing = torch.load(output_path, map_location="cpu")
        ensure_resume_compatible(existing, resume_compatibility, output_path)
        guided = existing["guided_samples"].detach().cpu()
        unguided = existing["unguided_samples"].detach().cpu()
        selected = existing["selected_particle_indices"].detach().cpu()
        boundary = existing["selected_boundary_indices"].detach().cpu()
        if guided.shape != unguided.shape or tuple(guided.shape[1:]) != sample_shape:
            raise ValueError("Resume file contains incompatible paired tensors.")
        if selected.shape != (guided.shape[0],) or boundary.shape != selected.shape:
            raise ValueError("Resume file contains incompatible lineage indices.")
        if guided.shape[0] > target_num_samples:
            raise ValueError("Resume file contains more pairs than requested.")
        num_completed = int(existing.get("num_completed", guided.shape[0]))
        if num_completed != guided.shape[0]:
            raise ValueError("Resume num_completed does not match saved pair count.")
        num_batches_completed = int(existing.get("num_batches_completed", 0))
        all_guided.append(guided)
        all_unguided.append(unguided)
        all_selected.append(selected)
        all_boundary.append(boundary)
        batch_resampling_steps.extend(existing.get("batch_resampling_steps", []))
        batch_replay_start_steps.extend(
            existing.get("batch_replay_start_steps", [])
        )
        batch_replay_num_steps.extend(existing.get("batch_replay_num_steps", []))
        print(f"Resuming lineage pairs: {num_completed}/{target_num_samples}")

    remaining = target_num_samples - num_completed
    while remaining > 0:
        batch_n = min(batch_size, remaining)
        batch_seed = base_seed + num_batches_completed
        sampler = LineageReplayInceptionTDSSampler(
            seed=batch_seed,
            num_particles=int(cfg.sampler.num_particles),
            sde=sde,
            num_samples=batch_n,
            sample_shape=sample_shape,
            ratio_potential=ratio_potential,
            num_steps=int(cfg.sampling.num_steps),
            guidance_scale=float(cfg.jeffrey.guidance_scale),
            resample_type=str(cfg.sampler.resample_type),
            guidance_ramp=cfg.sampler.guidance_ramp,
            guidance_start=float(cfg.sampler.guidance_start),
            adaptive_resampling=bool(cfg.sampler.adaptive_resampling),
            ess_threshold=float(cfg.sampler.ess_threshold),
            max_guidance_grad_norm=cfg.sampler.max_guidance_grad_norm,
        )
        result = sampler.sample(
            model,
            device,
            progress=bool(cfg.sampling.progress),
        )
        all_guided.append(result["guided_samples"].detach().cpu())
        all_unguided.append(result["unguided_samples"].detach().cpu())
        all_selected.append(result["selected_particle_indices"].detach().cpu())
        all_boundary.append(result["selected_boundary_indices"].detach().cpu())
        batch_resampling_steps.append(int(result["resampling_steps"]))
        batch_replay_start_steps.append(int(result["replay_start_step"]))
        batch_replay_num_steps.append(int(result["replay_num_steps"]))
        remaining -= batch_n
        num_completed += batch_n
        num_batches_completed += 1

        if resume_enabled and (
            num_batches_completed % save_every_batches == 0 or remaining == 0
        ):
            atomic_torch_save(
                make_payload(
                    guided_samples=torch.cat(all_guided),
                    unguided_samples=torch.cat(all_unguided),
                    selected_particle_indices=torch.cat(all_selected),
                    selected_boundary_indices=torch.cat(all_boundary),
                    batch_resampling_steps=batch_resampling_steps,
                    batch_replay_start_steps=batch_replay_start_steps,
                    batch_replay_num_steps=batch_replay_num_steps,
                    complete=False,
                    num_completed=num_completed,
                    num_batches_completed=num_batches_completed,
                    run_label=run_label,
                    ratio_classifier_path=ratio_classifier_path,
                    artifact_path=artifact_path,
                    requested_weight_type=requested_weight_type,
                    loaded_weight_type=loaded_weight_type,
                    cfg=cfg,
                    resume_compatibility=resume_compatibility,
                ),
                output_path,
            )
            print(f"Saved resumable lineage pairs: {num_completed}/{target_num_samples}")

    guided_samples = torch.cat(all_guided)
    unguided_samples = torch.cat(all_unguided)
    selected_indices = torch.cat(all_selected)
    boundary_indices = torch.cat(all_boundary)
    pixel_l2, pixel_mse = pixel_distances(guided_samples, unguided_samples)
    guided_logits = compute_ratio_logits(
        potential=ratio_potential,
        samples=guided_samples,
        device=device,
        batch_size=int(cfg.ratio.batch_size),
    )
    unguided_logits = compute_ratio_logits(
        potential=ratio_potential,
        samples=unguided_samples,
        device=device,
        batch_size=int(cfg.ratio.batch_size),
    )

    final_payload = make_payload(
        guided_samples=guided_samples,
        unguided_samples=unguided_samples,
        selected_particle_indices=selected_indices,
        selected_boundary_indices=boundary_indices,
        batch_resampling_steps=batch_resampling_steps,
        batch_replay_start_steps=batch_replay_start_steps,
        batch_replay_num_steps=batch_replay_num_steps,
        complete=True,
        num_completed=int(guided_samples.shape[0]),
        num_batches_completed=num_batches_completed,
        run_label=run_label,
        ratio_classifier_path=ratio_classifier_path,
        artifact_path=artifact_path,
        requested_weight_type=requested_weight_type,
        loaded_weight_type=loaded_weight_type,
        cfg=cfg,
        resume_compatibility=resume_compatibility,
        pixel_l2=pixel_l2,
        pixel_mse=pixel_mse,
        guided_ratio_logits=guided_logits,
        unguided_ratio_logits=unguided_logits,
    )
    atomic_torch_save(final_payload, output_path)
    metrics = save_analysis(
        output_dir=analysis_dir,
        guided_samples=guided_samples,
        unguided_samples=unguided_samples,
        pixel_l2=pixel_l2,
        pixel_mse=pixel_mse,
        guided_logits=guided_logits,
        unguided_logits=unguided_logits,
        top_k=top_k,
        preview_pairs=preview_pairs,
        pairing_semantics=PAIRING_SEMANTICS,
        pairing_caveat=PAIRING_CAVEAT,
    )

    print(f"Loaded artifact: {artifact_path}")
    print(f"Loaded weight type: {loaded_weight_type} (requested: {requested_weight_type})")
    print(f"Loaded ratio classifier: {ratio_classifier_path}")
    print(f"Saved lineage-paired payload to: {output_path}")
    print(f"Saved lineage-paired analysis to: {analysis_dir}")
    print(f"Pixel L2 mean: {metrics['pixel_l2_mean']:.6f}")
    print(f"Pixel L2 max: {metrics['pixel_l2_max']:.6f}")
    print(PAIRING_CAVEAT)


if __name__ == "__main__":
    main()
