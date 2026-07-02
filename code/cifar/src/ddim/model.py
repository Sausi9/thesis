from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"Expected timesteps with shape [batch], got {tuple(timesteps.shape)}.")

    half_dim = embedding_dim // 2
    scale = math.log(10000.0) / max(half_dim - 1, 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -scale
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = bool(with_conv)
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = bool(with_conv)
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.out_channels = out_channels
        self.use_conv_shortcut = bool(conv_shortcut)

        self.norm1 = normalize(self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    self.in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    self.in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.norm = normalize(self.in_channels)
        self.q = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h_size, w_size = q.shape
        q = q.reshape(b, c, h_size * w_size).permute(0, 2, 1)
        k = k.reshape(b, c, h_size * w_size)
        weights = torch.bmm(q, k) * (int(c) ** -0.5)
        weights = F.softmax(weights, dim=2)

        v = v.reshape(b, c, h_size * w_size)
        weights = weights.permute(0, 2, 1)
        h = torch.bmm(v, weights).reshape(b, c, h_size, w_size)
        h = self.proj_out(h)
        return x + h


class DDIMCIFARModel(nn.Module):
    """CIFAR-10 epsilon model matching the official ermongroup/ddim architecture."""

    def __init__(
        self,
        *,
        image_size: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        ch: int = 128,
        ch_mult: tuple[int, ...] | list[int] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] | list[int] = (16,),
        dropout: float = 0.1,
        resamp_with_conv: bool = True,
        num_diffusion_timesteps: int = 1000,
        model_type: str = "simple",
    ):
        super().__init__()
        if model_type not in {"simple", "bayesian"}:
            raise ValueError(f"Unsupported DDIM model_type '{model_type}'.")

        ch_mult = tuple(int(mult) for mult in ch_mult)
        attn_resolutions = {int(resolution) for resolution in attn_resolutions}
        self.ch = int(ch)
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.resolution = int(image_size)
        self.in_channels = int(in_channels)

        if model_type == "bayesian":
            self.logvar = nn.Parameter(torch.zeros(int(num_diffusion_timesteps)))

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                nn.Linear(self.ch, self.temb_ch),
                nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )

        self.conv_in = nn.Conv2d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = self.resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch * ch_mult[i_level]
            skip_in = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = self.ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, int(out_channels), kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != self.resolution or x.shape[3] != self.resolution:
            raise ValueError(
                f"Expected spatial shape {self.resolution}x{self.resolution}, got {tuple(x.shape[2:])}."
            )

        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)

