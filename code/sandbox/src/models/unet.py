import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embeddings, as used in DDPM-style models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.float()

        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype) * -emb_scale
        )
        angles = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Upsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1) if with_conv else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float,
        conv_shortcut: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            if conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.shortcut(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = _group_norm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).reshape(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        k = self.k(h_).reshape(b, c, h * w)  # [B, C, HW]
        v = self.v(h_).reshape(b, c, h * w).transpose(1, 2)  # [B, HW, C]

        attn = torch.bmm(q, k) * (c ** -0.5)  # [B, HW, HW]
        attn = torch.softmax(attn, dim=-1)
        h_ = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class UNet(nn.Module):
    """
    PyTorch rewrite of the original DDPM-style U-Net.

    Defaults are set for MNIST-like diffusion:
    input/output shape [B, 1, 28, 28] and timestep shape [B].
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (7,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
    ):
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(channel_mult)
        self.attn_resolutions = set(attn_resolutions)

        temb_channels = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(model_channels),
            nn.Linear(model_channels, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )

        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        hs_channels = [model_channels]
        ch = model_channels
        for i_level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResnetBlock(ch, out_ch, temb_channels, dropout))
                ch = out_ch
                level_attns.append(AttentionBlock(ch))
                hs_channels.append(ch)

            self.down_blocks.append(level_blocks)
            self.down_attns.append(level_attns)

            if i_level != self.num_resolutions - 1:
                self.downsamples.append(Downsample(ch, with_conv=resamp_with_conv))
                hs_channels.append(ch)
            else:
                self.downsamples.append(nn.Identity())

        self.mid_block1 = ResnetBlock(ch, ch, temb_channels, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResnetBlock(ch, ch, temb_channels, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i_level in reversed(range(self.num_resolutions)):
            out_ch = model_channels * channel_mult[i_level]
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = hs_channels.pop()
                level_blocks.append(ResnetBlock(ch + skip_ch, out_ch, temb_channels, dropout))
                ch = out_ch
                level_attns.append(AttentionBlock(ch))

            self.up_blocks.append(level_blocks)
            self.up_attns.append(level_attns)

            if i_level != 0:
                self.upsamples.append(Upsample(ch, with_conv=resamp_with_conv))
            else:
                self.upsamples.append(nn.Identity())

        self.norm_out = _group_norm(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t is None:
            raise ValueError("Timestep tensor t is required.")
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != x.shape[0]:
            raise ValueError("Batch size mismatch: x and t must have the same first dimension.")

        temb = self.time_embed(t)
        h = self.conv_in(x)
        hs = [h]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down_blocks[i_level][i_block](h, temb)
                if h.shape[-1] in self.attn_resolutions:
                    h = self.down_attns[i_level][i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = self.downsamples[i_level](h)
                hs.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                skip = hs.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[i_level][i_block](h, temb)
                if h.shape[-1] in self.attn_resolutions:
                    h = self.up_attns[i_level][i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.upsamples[i_level](h)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)
