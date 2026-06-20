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
        skip_rescale: bool = False,
    ):
        super().__init__()
        activation_layer = _activation_factory(activation)
        self.skip_rescale = bool(skip_rescale)

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
        out = h + self.skip(x)
        if self.skip_rescale:
            return out / math.sqrt(2.0)
        return out


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


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, skip_rescale: bool):
        super().__init__()
        self.channels = int(channels)
        self.skip_rescale = bool(skip_rescale)
        self.norm = _group_norm(self.channels)
        self.q = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.k = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.v = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normed = self.norm(x)
        q = self.q(normed).reshape(b, c, h * w)
        k = self.k(normed).reshape(b, c, h * w)
        v = self.v(normed).reshape(b, c, h * w)

        attn = torch.einsum("bcn,bcm->bnm", q, k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bnm,bcm->bcn", attn, v).reshape(b, c, h, w)
        out = self.proj(out)
        if self.skip_rescale:
            return (x + out) / math.sqrt(2.0)
        return x + out


class ResBlockWithAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float,
        activation: str,
        skip_rescale: bool,
        use_attention: bool,
    ):
        super().__init__()
        self.block = ResidualBlock(
            in_channels,
            out_channels,
            time_dim,
            dropout,
            activation,
            skip_rescale=skip_rescale,
        )
        self.attention = (
            AttentionBlock(out_channels, skip_rescale=skip_rescale)
            if use_attention
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block(x, t_emb)
        return self.attention(h)


class ScoreUNet(nn.Module):
    """Stronger CIFAR U-Net with configurable depth and spatial attention."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        image_size: int = 32,
        model_channels: int = 128,
        channel_mult: list[int] | tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: list[int] | tuple[int, ...] = (16,),
        time_embed_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "silu",
        skip_rescale: bool = True,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if len(channel_mult) < 2:
            raise ValueError("ScoreUNet expects at least two channel multipliers.")
        if int(num_res_blocks) <= 0:
            raise ValueError(f"num_res_blocks must be positive, got {num_res_blocks}.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.image_size = int(image_size)
        self.model_channels = int(model_channels)
        self.channel_mult = tuple(int(mult) for mult in channel_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attention_resolutions = {
            int(resolution) for resolution in attention_resolutions
        }
        self.skip_rescale = bool(skip_rescale)

        activation_layer = _activation_factory(activation)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            activation_layer(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        channels = [self.model_channels * mult for mult in self.channel_mult]
        self.input = nn.Conv2d(self.in_channels, channels[0], kernel_size=3, padding=1)

        current_channels = channels[0]
        current_resolution = self.image_size
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: list[int] = []

        for level, out_channels_level in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(
                    ResBlockWithAttention(
                        current_channels,
                        out_channels_level,
                        time_embed_dim,
                        dropout,
                        activation,
                        skip_rescale=self.skip_rescale,
                        use_attention=current_resolution in self.attention_resolutions,
                    )
                )
                current_channels = out_channels_level
            self.down_blocks.append(blocks)
            self.skip_channels.append(current_channels)

            if level != len(channels) - 1:
                self.downsamples.append(Downsample(current_channels))
                current_resolution //= 2

        self.mid1 = ResBlockWithAttention(
            current_channels,
            current_channels,
            time_embed_dim,
            dropout,
            activation,
            skip_rescale=self.skip_rescale,
            use_attention=True,
        )
        self.mid2 = ResBlockWithAttention(
            current_channels,
            current_channels,
            time_embed_dim,
            dropout,
            activation,
            skip_rescale=self.skip_rescale,
            use_attention=False,
        )

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(channels) - 1)):
            current_resolution *= 2
            self.upsamples.append(Upsample(current_channels))
            blocks = nn.ModuleList()
            skip_channels = self.skip_channels[level]
            out_channels_level = channels[level]
            blocks.append(
                ResBlockWithAttention(
                    current_channels + skip_channels,
                    out_channels_level,
                    time_embed_dim,
                    dropout,
                    activation,
                    skip_rescale=self.skip_rescale,
                    use_attention=current_resolution in self.attention_resolutions,
                )
            )
            current_channels = out_channels_level
            for _ in range(self.num_res_blocks - 1):
                blocks.append(
                    ResBlockWithAttention(
                        current_channels,
                        current_channels,
                        time_embed_dim,
                        dropout,
                        activation,
                        skip_rescale=self.skip_rescale,
                        use_attention=current_resolution in self.attention_resolutions,
                    )
                )
            self.up_blocks.append(blocks)

        self.output = nn.Sequential(
            _group_norm(current_channels),
            activation_layer(),
            nn.Conv2d(current_channels, self.out_channels, kernel_size=3, padding=1),
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
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected spatial shape {(self.image_size, self.image_size)}, "
                f"got {tuple(x.shape[-2:])}."
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
        h = self.input(x)

        skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.mid2(self.mid1(h, t_emb), t_emb)

        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, reversed(skips[:-1])):
            h = upsample(h)
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)

        return self.output(h)
