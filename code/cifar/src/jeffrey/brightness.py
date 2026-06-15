import torch


def brightness(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected image tensor [B, C, H, W], got {tuple(x.shape)}.")
    return ((x + 1.0) / 2.0).mean(dim=(1, 2, 3))
