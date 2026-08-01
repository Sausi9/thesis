from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CMMDClipEncoder:
    """OpenAI CLIP image encoder compatible with the reference CMMD setup."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_name: str = "ViT-L-14-336",
        pretrained: str = "openai",
        cache_dir: str | Path | None = None,
    ):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "CMMD evaluation requires open-clip-torch. Run `uv sync` first."
            ) from exc

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
        model.eval()
        model.requires_grad_(False)

        visual_size = model.visual.image_size
        if isinstance(visual_size, (tuple, list)):
            if len(visual_size) != 2 or int(visual_size[0]) != int(visual_size[1]):
                raise ValueError(f"Expected a square CLIP image size, got {visual_size}.")
            visual_size = visual_size[0]

        self.model = model
        self.device = device
        self.image_size = int(visual_size)
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError(
                f"Expected CIFAR images [B, 3, 32, 32], got {tuple(images.shape)}."
            )
        images = ((images.to(self.device).float().clamp(-1, 1) + 1.0) / 2.0)
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        )
        images = (images - self.mean) / self.std
        return self.model.encode_image(images, normalize=True).float()


@torch.no_grad()
def extract_clip_embeddings(
    *,
    encoder: CMMDClipEncoder,
    dataloader: Iterable,
    max_samples: int,
) -> torch.Tensor:
    embeddings = []
    seen = 0
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        remaining = int(max_samples) - seen
        if remaining <= 0:
            break
        batch = batch[:remaining]
        embeddings.append(encoder.encode(batch).detach().cpu())
        seen += int(batch.shape[0])

    if seen != int(max_samples):
        raise ValueError(f"Expected {max_samples} images, but only encoded {seen}.")
    return torch.cat(embeddings, dim=0).float()


def _kernel_mean(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sigma: float,
    batch_size: int,
) -> float:
    gamma = 1.0 / (2.0 * float(sigma) ** 2)
    total = 0.0
    count = int(x.shape[0]) * int(y.shape[0])

    for x_start in range(0, int(x.shape[0]), int(batch_size)):
        x_batch = x[x_start : x_start + int(batch_size)]
        x_sqnorm = x_batch.square().sum(dim=1, keepdim=True)
        for y_start in range(0, int(y.shape[0]), int(batch_size)):
            y_batch = y[y_start : y_start + int(batch_size)]
            y_sqnorm = y_batch.square().sum(dim=1).unsqueeze(0)
            squared_distance = (
                x_sqnorm + y_sqnorm - 2.0 * x_batch @ y_batch.transpose(0, 1)
            ).clamp_min_(0.0)
            block_sum = torch.exp(-gamma * squared_distance).sum(dtype=torch.float64)
            total += float(block_sum)
    return total / count


@torch.no_grad()
def cmmd(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    sigma: float = 10.0,
    scale: float = 1000.0,
    batch_size: int = 1024,
) -> float:
    """Google CMMD's scaled, biased Gaussian-kernel MMD estimator."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Expected embeddings [N, D] and [M, D], got {tuple(x.shape)} and {tuple(y.shape)}."
        )
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("CMMD requires two non-empty embedding sets.")
    if float(sigma) <= 0.0 or int(batch_size) <= 0:
        raise ValueError("sigma and batch_size must be positive.")

    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    k_xx = _kernel_mean(x, x, sigma=float(sigma), batch_size=int(batch_size))
    k_xy = _kernel_mean(x, y, sigma=float(sigma), batch_size=int(batch_size))
    k_yy = _kernel_mean(y, y, sigma=float(sigma), batch_size=int(batch_size))
    return float(scale) * (k_xx + k_yy - 2.0 * k_xy)
