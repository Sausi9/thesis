from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from torchvision.utils import make_grid, save_image

from src.ddim.checkpoint import resolve_ddim_checkpoint
from src.ddim.model import DDIMCIFARModel
from src.ddim.sampler import ddim_sample_batch, make_linear_beta_schedule
from src.jeffrey.brightness import brightness
from src.utils import (
    find_latest_artifact,
    load_model_state,
    resolve_device,
    resolve_path,
    timestamped_output_path,
)


def make_output_path(cfg: DictConfig, project_root: Path, run_name: str) -> Path:
    return timestamped_output_path(
        output_dir=resolve_path(project_root, str(cfg.sampling.output_dir)),
        output_name=cfg.sampling.output_name,
        default_stem=f"{run_name}_model_samples",
        extension=".pt",
    )


def make_preview_path(cfg: DictConfig, project_root: Path, output_path: Path) -> Path:
    preview_dir = resolve_path(project_root, str(cfg.sampling.preview_dir))
    preview_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir / f"{output_path.stem}.png"


@torch.no_grad()
def euler_maruyama_sample(
    *,
    model,
    sde,
    num_samples: int,
    sample_shape: tuple[int, int, int],
    num_steps: int,
    device: torch.device,
    return_mean: bool,
    progress: bool,
) -> torch.Tensor:
    model.eval()
    x = sde.prior_sample((num_samples, *sample_shape), device)
    timesteps = torch.linspace(
        float(sde.config.t_max),
        float(sde.config.t_min),
        num_steps + 1,
        device=device,
    )

    iterator = range(num_steps)
    if progress:
        iterator = tqdm(iterator, desc="Sampling", leave=True)

    for i in iterator:
        t = timesteps[i]
        t_next = timesteps[i + 1]
        step_size = t - t_next
        t_batch = torch.full((num_samples,), t, device=device)

        score = model(x, t_batch)
        mean, variance = sde.reverse_transition_params(x, t_batch, score, step_size)
        std = torch.sqrt(variance).view(num_samples, 1, 1, 1)

        if i == num_steps - 1 and return_mean:
            x = mean
        else:
            x = mean + std * torch.randn_like(x)

    return x


def save_preview(cfg: DictConfig, project_root: Path, output_path: Path, samples: torch.Tensor) -> None:
    if not bool(cfg.sampling.save_preview):
        return
    preview_count = min(int(cfg.sampling.preview_num_samples), samples.shape[0])
    preview = (samples[:preview_count].clamp(-1, 1) + 1) / 2
    grid = make_grid(preview, nrow=int(cfg.sampling.preview_nrow))
    preview_path = make_preview_path(cfg, project_root, output_path)
    save_image(grid, preview_path)
    print(f"Saved preview to: {preview_path}")


def run_score_sde_sample(cfg: DictConfig, project_root: Path, device: torch.device) -> None:
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
    sde = instantiate(cfg.sde)

    sample_shape = tuple(int(v) for v in cfg.dataset.shape)
    num_samples = int(cfg.sampling.num_samples)
    batch_size = int(cfg.sampling.batch_size)
    batches = []
    remaining = num_samples
    while remaining > 0:
        batch_n = min(batch_size, remaining)
        samples_batch = euler_maruyama_sample(
            model=model,
            sde=sde,
            num_samples=batch_n,
            sample_shape=sample_shape,
            num_steps=int(cfg.sampling.num_steps),
            device=device,
            return_mean=bool(cfg.sampling.return_mean),
            progress=bool(cfg.sampling.progress),
        )
        batches.append(samples_batch.detach().cpu())
        remaining -= batch_n

    samples = torch.cat(batches, dim=0)
    brightness_values = brightness(samples)
    run_name = str(payload.get("run_name") or artifact_path.stem)
    output_path = make_output_path(cfg, project_root, run_name)
    result = {
        "samples": samples,
        "sample_type": "model",
        "artifact_path": str(artifact_path),
        "requested_weight_type": requested_weight_type,
        "loaded_weight_type": loaded_weight_type,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "image_shape": sample_shape,
        "brightness_mean": brightness_values.mean().detach().cpu(),
        "brightness_std": brightness_values.std(
            unbiased=brightness_values.numel() > 1
        ).detach().cpu(),
    }
    torch.save(result, output_path)
    save_preview(cfg, project_root, output_path, samples)

    print("Sampling backend: score_sde")
    print(f"Loaded artifact: {artifact_path}")
    print(f"Loaded weight type: {loaded_weight_type} (requested: {requested_weight_type})")
    print(f"Saved samples to: {output_path}")
    print(f"Brightness mean: {float(result['brightness_mean']):.6f}")
    print(f"Brightness std: {float(result['brightness_std']):.6f}")


def run_ddim_sample(cfg: DictConfig, project_root: Path, device: torch.device) -> None:
    checkpoint_path_cfg = cfg.ddim.checkpoint_path
    checkpoint_path = (
        resolve_path(project_root, str(checkpoint_path_cfg))
        if checkpoint_path_cfg is not None
        else None
    )
    cache_dir_cfg = cfg.ddim.cache_dir
    cache_dir = (
        resolve_path(project_root, str(cache_dir_cfg))
        if cache_dir_cfg is not None
        else None
    )
    checkpoint_path = resolve_ddim_checkpoint(
        checkpoint_name=str(cfg.ddim.checkpoint_name),
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        download_enabled=bool(cfg.ddim.download),
        check_md5=bool(cfg.ddim.check_md5),
    )

    model = DDIMCIFARModel(
        image_size=int(cfg.ddim.image_size),
        in_channels=int(cfg.ddim.in_channels),
        out_channels=int(cfg.ddim.out_channels),
        ch=int(cfg.ddim.ch),
        ch_mult=tuple(int(v) for v in cfg.ddim.ch_mult),
        num_res_blocks=int(cfg.ddim.num_res_blocks),
        attn_resolutions=tuple(int(v) for v in cfg.ddim.attn_resolutions),
        dropout=float(cfg.ddim.dropout),
        resamp_with_conv=bool(cfg.ddim.resamp_with_conv),
        num_diffusion_timesteps=int(cfg.ddim.num_diffusion_timesteps),
        model_type=str(cfg.ddim.model_type),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    betas = make_linear_beta_schedule(
        num_timesteps=int(cfg.ddim.num_diffusion_timesteps),
        beta_start=float(cfg.ddim.beta_start),
        beta_end=float(cfg.ddim.beta_end),
        device=device,
    )

    sample_shape = tuple(int(v) for v in cfg.dataset.shape)
    num_samples = int(cfg.sampling.num_samples)
    batch_size = int(cfg.ddim.batch_size)
    batches = []
    remaining = num_samples
    while remaining > 0:
        batch_n = min(batch_size, remaining)
        samples_batch = ddim_sample_batch(
            model=model,
            sample_shape=(batch_n, *sample_shape),
            betas=betas,
            timesteps=int(cfg.ddim.timesteps),
            eta=float(cfg.ddim.eta),
            skip_type=str(cfg.ddim.skip_type),
            num_diffusion_timesteps=int(cfg.ddim.num_diffusion_timesteps),
            device=device,
            progress=bool(cfg.sampling.progress),
        )
        batches.append(samples_batch.detach().cpu())
        remaining -= batch_n

    samples = torch.cat(batches, dim=0)
    brightness_values = brightness(samples)
    run_name = f"ddim_{cfg.ddim.checkpoint_name}"
    output_path = make_output_path(cfg, project_root, run_name)
    result = {
        "samples": samples,
        "sample_type": "ddim_model",
        "backend": "ddim",
        "ddim_checkpoint_name": str(cfg.ddim.checkpoint_name),
        "ddim_checkpoint_path": str(checkpoint_path),
        "ddim_timesteps": int(cfg.ddim.timesteps),
        "ddim_eta": float(cfg.ddim.eta),
        "ddim_skip_type": str(cfg.ddim.skip_type),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "image_shape": sample_shape,
        "brightness_mean": brightness_values.mean().detach().cpu(),
        "brightness_std": brightness_values.std(
            unbiased=brightness_values.numel() > 1
        ).detach().cpu(),
    }
    torch.save(result, output_path)
    save_preview(cfg, project_root, output_path, samples)

    print("Sampling backend: ddim")
    print(f"Loaded DDIM checkpoint: {checkpoint_path}")
    print(f"DDIM timesteps: {int(cfg.ddim.timesteps)}")
    print(f"DDIM eta: {float(cfg.ddim.eta):g}")
    print(f"Saved samples to: {output_path}")
    print(f"Brightness mean: {float(result['brightness_mean']):.6f}")
    print(f"Brightness std: {float(result['brightness_std']):.6f}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))

    backend = str(OmegaConf.select(cfg, "sampling.backend", default="score_sde"))
    if backend == "score_sde":
        run_score_sde_sample(cfg, project_root, device)
        return
    if backend == "ddim":
        run_ddim_sample(cfg, project_root, device)
        return
    raise ValueError("sampling.backend must be one of: score_sde, ddim.")


if __name__ == "__main__":
    main()
