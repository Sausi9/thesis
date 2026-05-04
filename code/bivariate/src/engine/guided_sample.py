from datetime import datetime
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.engine.train import resolve_device
from src.samplers.reverse_guided import GuidedReverseSDESampler
from src.distributions.targets import GaussianTargetMarginal


def resolve_path(project_root: Path, path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


def find_latest_artifact(artifact_dir: Path, preference: str) -> Path:
    patterns = {
        "best": "*_best.pt",
        "final": "*_final.pt",
        "any": "*.pt",
    }
    if preference not in patterns:
        valid = ", ".join(sorted(patterns))
        raise ValueError(f"Unknown artifact_preference '{preference}'. Expected one of: {valid}.")

    matches = sorted(
        artifact_dir.glob(patterns[preference]),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No artifacts matching {patterns[preference]} in {artifact_dir}.")
    return matches[0]


def load_model_state(payload: dict) -> dict:
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    raise KeyError("Expected artifact/checkpoint to contain 'model_state_dict'.")


def make_output_path(cfg: DictConfig, project_root: Path, run_name: str) -> Path:
    output_dir = resolve_path(project_root, str(cfg.sampling.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.sampling.output_name is not None:
        output_name = str(cfg.sampling.output_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{run_name}_reverse_guided_samples_{timestamp}.pt"

    if not output_name.endswith(".pt"):
        output_name = f"{output_name}.pt"
    return output_dir / output_name



@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
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
    dim = int(cfg.dataset.dim)
    
    #TODO: Finish the guided sampler
    #TODO: Just use torch.distributions.Normal instead of this useless GaussianTargetMarginal class
    target_mean = cfg.jeffrey.target.mean
    target_var = cfg.jeffrey.target.variance
    target_marginal = GaussianTargetMarginal(target_mean, target_var)

    updated_dim = cfg.jeffrey.updated_dim
    original_mean = cfg.dataset.mean[updated_dim]
    original_var = cfg.dataset.covariance[updated_dim][updated_dim]
    original_marginal = torch.distributions.Normal(original_mean, original_var ** 0.5)

    sampler = GuidedReverseSDESampler(sde, target_marginal, original_marginal, dim, updated_dim, num_samples)

    samples = sampler.sample(model, cfg.sampling.num_steps, device, True, True)
    run_name = str(payload.get("run_name") or artifact_path.stem)
    output_path = make_output_path(cfg, project_root, run_name)
    result = {
        "samples": samples.detach().cpu(),
        "artifact_path": str(artifact_path),
        "config": OmegaConf.to_container(cfg, resolve=True),
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
