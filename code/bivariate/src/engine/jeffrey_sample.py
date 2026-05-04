from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.distributions.gaussian import calculate_conditional_params
from src.distributions.targets import GaussianTargetMarginal
from src.jeffrey.update import (
    jeffrey_updated_gaussian_params,
    sample_jeffrey_update,
)


def build_target_marginal(target_cfg: DictConfig) -> GaussianTargetMarginal:
    target_type = str(target_cfg.type)
    if target_type != "gaussian":
        raise ValueError(f"Unsupported Jeffrey target type: {target_type}")

    return GaussianTargetMarginal(
        mean=float(target_cfg.mean),
        variance=float(target_cfg.variance),
    )


def make_output_path(cfg: DictConfig, project_root: Path) -> Path:
    output_dir = project_root / str(cfg.sampling.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.sampling.output_name is not None:
        output_name = str(cfg.sampling.output_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{cfg.dataset.name}_jeffrey_samples_{timestamp}.pt"

    if not output_name.endswith(".pt"):
        output_name = f"{output_name}.pt"
    return output_dir / output_name


# THIS FILE USES THE ANALYTIC FORM OF THE UPDATED JOINT, I.E THE CONDITIONAL TIMES THE NEW MARGINAL.
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not bool(cfg.jeffrey.enabled):
        raise ValueError("Jeffrey sampling requires jeffrey.enabled=true")

    project_root = Path(__file__).resolve().parents[2]
    torch.manual_seed(int(cfg.seed))

    joint_mean = cfg.dataset.mean
    joint_covariance = cfg.dataset.covariance
    updated_dim = int(cfg.jeffrey.updated_dim)
    target_marginal = build_target_marginal(cfg.jeffrey.target)

    updated_mean, updated_covariance = jeffrey_updated_gaussian_params(
        joint_mean=joint_mean,
        joint_covariance=joint_covariance,
        updated_dim=updated_dim,
        target_marginal=target_marginal,
    )
    samples = sample_jeffrey_update(
        joint_mean=joint_mean,
        joint_covariance=joint_covariance,
        updated_dim=updated_dim,
        target_marginal=target_marginal,
        num_samples=int(cfg.sampling.num_samples),
    )

    kept_dim = 1 - updated_dim
    conditional = calculate_conditional_params(
        joint_mean=joint_mean,
        joint_covariance=joint_covariance,
        marginal_dim=kept_dim,
        given_dim=updated_dim,
    )
    sample_mean = samples.mean(dim=0)
    sample_covariance = torch.cov(samples.T)
    output_path = make_output_path(cfg, project_root)

    result = {
        "samples": samples,
        "sample_type": "jeffrey_exact",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "updated_dim": updated_dim,
        "target_marginal": {
            "type": "gaussian",
            "mean": target_marginal.mean,
            "variance": target_marginal.variance,
        },
        "conditional_params": conditional,
        "updated_mean": updated_mean,
        "updated_covariance": updated_covariance,
        "sample_mean": sample_mean,
        "sample_covariance": sample_covariance,
    }
    torch.save(result, output_path)

    print(f"Saved Jeffrey samples to: {output_path}")
    print(
        f"Preserved conditional: dim {kept_dim} | dim {updated_dim}=z "
        f"~ N({conditional['intercept']:.4f} + "
        f"{conditional['slope']:.4f}z, {conditional['variance']:.4f})"
    )
    print(f"Updated analytic mean: {updated_mean.tolist()}")
    print(f"Updated analytic covariance: {updated_covariance.tolist()}")
    print(f"Sample mean: {sample_mean.tolist()}")
    print(f"Sample covariance: {sample_covariance.tolist()}")


if __name__ == "__main__":
    main()
