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


from omegaconf import DictConfig, OmegaConf


def make_output_path(cfg: DictConfig, project_root: Path, run_name: str) -> Path:
    return timestamped_output_path(
        output_dir=resolve_path(project_root, str(cfg.sampling.output_dir)),
        output_name=cfg.sampling.output_name,
        default_stem=f"{run_name}_tds_samples",
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
    sde = instantiate(cfg.sde)

    num_samples = int(cfg.sampling.num_samples)
    data_dim = int(cfg.dataset.dim)
    updated_dim = cfg.jeffrey.updated_dim

    target_marginal = build_target_marginal(cfg.jeffrey.target)

    original_mean = cfg.dataset.mean[updated_dim]
    original_var = cfg.dataset.covariance[updated_dim][updated_dim]
    original_marginal = torch.distributions.Normal(original_mean, original_var**0.5)

    updated_mean, updated_covariance = jeffrey_updated_gaussian_params(
        joint_mean=cfg.dataset.mean,
        joint_covariance=cfg.dataset.covariance,
        updated_dim=updated_dim,
        target_marginal=target_marginal,
    )
    
    tds = TDSSampler(
        cfg.sampler.num_particles,
        sde,
        num_samples,
        target_marginal,
        original_marginal,
        updated_dim,
        data_dim,
        cfg.sampling.num_steps,
    )
    samples = tds.sample(model, device)

    
    run_name = str(payload.get("run_name") or artifact_path.stem)
    output_path = make_output_path(cfg, project_root, run_name)
    result = {
        "samples": samples.detach().cpu(),
        "sample_type": "tds",
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

    print(f"Loaded artifact: {artifact_path}")
    print(f"Saved samples to: {output_path}")
    print(f"Sample mean: {result['sample_mean'].tolist()}")
    print(f"Sample covariance: {result['sample_covariance'].tolist()}")

if __name__ == "__main__":
    main()
