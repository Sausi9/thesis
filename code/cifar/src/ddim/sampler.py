from __future__ import annotations

import torch
from tqdm.auto import tqdm


def make_linear_beta_schedule(
    *,
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.linspace(float(beta_start), float(beta_end), int(num_timesteps), device=device)


def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    beta = torch.cat([torch.zeros(1, device=beta.device, dtype=beta.dtype), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


def make_ddim_sequence(*, num_timesteps: int, timesteps: int, skip_type: str) -> list[int]:
    num_timesteps = int(num_timesteps)
    timesteps = int(timesteps)
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}.")
    if timesteps > num_timesteps:
        raise ValueError(
            f"timesteps cannot exceed num_diffusion_timesteps ({num_timesteps}), got {timesteps}."
        )

    if skip_type == "uniform":
        skip = num_timesteps // timesteps
        return list(range(0, num_timesteps, skip))
    if skip_type == "quad":
        values = torch.linspace(0, (num_timesteps * 0.8) ** 0.5, timesteps) ** 2
        return [int(value.item()) for value in values]
    raise ValueError("skip_type must be one of: uniform, quad.")


@torch.no_grad()
def ddim_sample_batch(
    *,
    model,
    sample_shape: tuple[int, int, int, int],
    betas: torch.Tensor,
    timesteps: int,
    eta: float,
    skip_type: str,
    num_diffusion_timesteps: int,
    device: torch.device,
    progress: bool,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(sample_shape, device=device)
    sequence = make_ddim_sequence(
        num_timesteps=num_diffusion_timesteps,
        timesteps=timesteps,
        skip_type=skip_type,
    )
    sequence_next = [-1] + list(sequence[:-1])
    iterator = list(zip(reversed(sequence), reversed(sequence_next)))
    if progress:
        iterator = tqdm(iterator, desc="DDIM sampling", leave=True)

    n = int(sample_shape[0])
    for current_step, next_step in iterator:
        t = torch.full((n,), int(current_step), device=device, dtype=torch.long)
        next_t = torch.full((n,), int(next_step), device=device, dtype=torch.long)
        alpha_t = compute_alpha(betas, t)
        alpha_next = compute_alpha(betas, next_t)

        predicted_noise = model(x, t.float())
        predicted_x0 = (x - predicted_noise * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
        c1 = float(eta) * ((1 - alpha_t / alpha_next) * (1 - alpha_next) / (1 - alpha_t)).sqrt()
        c2 = ((1 - alpha_next) - c1**2).sqrt()
        x = alpha_next.sqrt() * predicted_x0 + c1 * torch.randn_like(x) + c2 * predicted_noise

    return x

