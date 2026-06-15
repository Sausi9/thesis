import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [batch], got {tuple(t.shape)}.")

        t = t.float()
        half_dim = self.dim // 2
        if half_dim == 0:
            return t[:, None]

        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype) * -scale
        )
        angles = t[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    activations: dict[str, Callable[[], nn.Module]] = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(activations))
        raise ValueError(f"Unknown activation '{name}'. Expected one of: {valid}.") from exc


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        activation_layer = _activation_factory(activation)

        self.norm1 = _group_norm(in_channels)
        self.act1 = activation_layer()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.act2 = activation_layer()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ScoreUNet(nn.Module):
    """Small time-conditioned U-Net that predicts VP-SDE scores for 32x32 images."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        channel_mult: list[int] | tuple[int, int, int] = (1, 2, 4),
        time_embed_dim: int = 256,
        dropout: float = 0.0,
        activation: str = "silu",
        zero_init_output: bool = True,
    ):
        super().__init__()
        if len(channel_mult) != 3:
            raise ValueError("ScoreUNet expects exactly three channel multipliers.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.model_channels = int(model_channels)
        self.channel_mult = tuple(int(mult) for mult in channel_mult)

        c1, c2, c3 = [self.model_channels * mult for mult in self.channel_mult]
        activation_layer = _activation_factory(activation)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            activation_layer(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input = nn.Conv2d(self.in_channels, c1, kernel_size=3, padding=1)
        self.enc1 = ResidualBlock(c1, c1, time_embed_dim, dropout, activation)
        self.down1 = Downsample(c1)
        self.enc2 = ResidualBlock(c1, c2, time_embed_dim, dropout, activation)
        self.down2 = Downsample(c2)
        self.enc3 = ResidualBlock(c2, c3, time_embed_dim, dropout, activation)

        self.mid1 = ResidualBlock(c3, c3, time_embed_dim, dropout, activation)
        self.mid2 = ResidualBlock(c3, c3, time_embed_dim, dropout, activation)

        self.up2 = Upsample(c3)
        self.dec2 = ResidualBlock(c3 + c2, c2, time_embed_dim, dropout, activation)
        self.up1 = Upsample(c2)
        self.dec1 = ResidualBlock(c2 + c1, c1, time_embed_dim, dropout, activation)

        self.output = nn.Sequential(
            _group_norm(c1),
            activation_layer(),
            nn.Conv2d(c1, self.out_channels, kernel_size=3, padding=1),
        )

        if zero_init_output:
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [batch], got {tuple(t.shape)}.")
        if t.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x has {x.shape[0]}, t has {t.shape[0]}."
            )

        t_emb = self.time_embed(t.to(device=x.device, dtype=x.dtype))

        h0 = self.input(x)
        h1 = self.enc1(h0, t_emb)
        h2 = self.enc2(self.down1(h1), t_emb)
        h3 = self.enc3(self.down2(h2), t_emb)

        h = self.mid2(self.mid1(h3, t_emb), t_emb)
        h = self.up2(h)
        h = self.dec2(torch.cat([h, h2], dim=1), t_emb)
        h = self.up1(h)
        h = self.dec1(torch.cat([h, h1], dim=1), t_emb)
        return self.output(h)
