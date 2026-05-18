from matplotlib.pylab import mean
from pathlib import Path

import torch
import hydra

from hydra.utils import instantiate

from src.distributions.targets import build_target_marginal
from src.jeffrey.update import jeffrey_updated_gaussian_params
from src.utils import (
    find_latest_artifact,
    load_model_state,
    resolve_device,
    resolve_path,
    timestamped_output_path,
)

from src.samplers.tds import TDSSampler
from src.distributions.targets import estimate_model_marginal
from src.distributions.gaussian import estimate_model_gaussian_params


from omegaconf import DictConfig, OmegaConf

def load_samples(sample_path):
    payload = torch.load(sample_path, map_location="cpu")
    if payload.get("sample_type") != "model":
        raise ValueError(
            f"Expected unconditional model samples, got {payload.get('sample_type')}"
        )
    return payload["samples"]

def make_run_label(
    twist_type: str,
    guidance_ramp: str,
    resample_type: str,
    adaptive_resampling: bool,
    ess_threshold: float,
    num_particles: int,
    num_steps: int,
) -> str:
    mode = "adaptive" if adaptive_resampling else "always"
    threshold = f"_ess{ess_threshold:g}" if adaptive_resampling else ""
    ramp = "" if guidance_ramp == "none" else f"_{guidance_ramp}-ramp"
    return f"{twist_type}{ramp}_{resample_type}_{mode}{threshold}_K{num_particles}_T{num_steps}"


def make_output_path(cfg: DictConfig, project_root: Path, run_name: str, run_label: str) -> Path:
    return timestamped_output_path(
        output_dir=resolve_path(project_root, str(cfg.sampling.output_dir)),
        output_name=cfg.sampling.output_name,
        default_stem=f"{run_name}_tds_samples_{run_label}",
        extension=".pt",
    )

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))

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
    model.load_state_dict(load_model_state(payload))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    sde = instantiate(cfg.sde)

    num_samples = int(cfg.sampling.num_samples)
    data_dim = int(cfg.dataset.dim)
    updated_dim = cfg.jeffrey.updated_dim

    target_marginal = build_target_marginal(cfg.jeffrey.target)
    samples = load_samples(cfg.jeffrey.source_sample_path)
    # "original marginal" is the model induced marginal here, not the exact original marginal. TDS does this in the paper
    original_marginal = estimate_model_marginal(samples, updated_dim)


    # original_mean = cfg.dataset.mean[updated_dim]
    # original_var = cfg.dataset.covariance[updated_dim][updated_dim]
    # original_marginal = torch.distributions.Normal(original_mean, original_var**0.5)

    original_mean, original_cov = estimate_model_gaussian_params(samples)

    updated_mean, updated_covariance = jeffrey_updated_gaussian_params(
        joint_mean=cfg.dataset.mean,
        joint_covariance=cfg.dataset.covariance,
        updated_dim=updated_dim,
        target_marginal=target_marginal,
    )

    twist_type = str(cfg.sampler.twist_type)
    guidance_ramp = str(cfg.sampler.guidance_ramp)
    resample_type = str(cfg.sampler.resample_type)
    adaptive_resampling = bool(cfg.sampler.adaptive_resampling)
    ess_threshold = float(cfg.sampler.ess_threshold)
    num_particles = int(cfg.sampler.num_particles)
    num_steps = int(cfg.sampling.num_steps)
    run_label = make_run_label(
        twist_type,
        guidance_ramp,
        resample_type,
        adaptive_resampling,
        ess_threshold,
        num_particles,
        num_steps,
    )

    all_samples = []
    remaining_samples = num_samples
    while remaining_samples > 0:
        batch_n = min(cfg.sampling.batch_size, remaining_samples)
        tds = TDSSampler(
            num_particles,
            sde,
            batch_n,
            target_marginal,
            original_marginal,
            updated_dim,
            data_dim,
            num_steps,
            twist_type=twist_type,
            base_mean=original_mean,
            base_covariance=original_cov,
            updated_mean=updated_mean,
            updated_covariance=updated_covariance,
            resample_type=resample_type,
            guidance_ramp=guidance_ramp,
            adaptive_resampling=adaptive_resampling,
            ess_threshold=ess_threshold,
        )
        samples_batch = tds.sample(model, device, progress=bool(cfg.sampling.progress))
        all_samples.append(samples_batch.detach().cpu())
        remaining_samples -= batch_n
    samples = torch.cat(all_samples, dim=0)
    
    run_name = str(payload.get("run_name") or artifact_path.stem)
    output_path = make_output_path(cfg, project_root, run_name, run_label)
    result = {
        "samples": samples.detach().cpu(),
        "sample_type": "tds",
        "run_label": run_label,
        "twist_type": twist_type,
        "guidance_ramp": guidance_ramp,
        "resample_type": resample_type,
        "adaptive_resampling": adaptive_resampling,
        "ess_threshold": ess_threshold,
        "num_particles": num_particles,
        "num_steps": num_steps,
        "seed": int(cfg.seed),
        "artifact_path": str(artifact_path),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "updated_dim": updated_dim,
        "target_marginal": {
            "type": "gaussian",
            "mean": target_marginal.mean,
            "variance": target_marginal.variance,
        },
        "updated_mean": updated_mean,
        "updated_covariance": updated_covariance,
        "sample_mean": samples.mean(dim=0).detach().cpu(),
        "sample_covariance": torch.cov(samples.T.detach().cpu()),
    }
    torch.save(result, output_path)

    mean_diff = result['sample_mean'] - result['updated_mean']
    cov_diff = result['sample_covariance'] - result['updated_covariance']
    
    print(f"Loaded artifact: {artifact_path}")
    print(f"Saved samples to: {output_path}")
    print(f"Sample mean: {result['sample_mean'].tolist()}")
    print(f"Sample covariance: {result['sample_covariance'].tolist()}")
    print(f"Mean diff: {mean_diff.tolist()}")
    print(f"Cov diff: {cov_diff}")

if __name__ == "__main__":
    main()
