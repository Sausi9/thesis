import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar diffusion times."""

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


class ScoreMLP(nn.Module):
    """Time-conditioned MLP for 2D continuous-time score estimation.

    The model predicts the score directly:

        s_theta(x_t, t) ~= grad_x log p_t(x_t)

    With VP-SDE noising, the denoising score-matching target is
    -epsilon / std. This keeps the training target equivalent to the common
    DDPM epsilon target while exposing score outputs for reverse SDE sampling
    and Jeffrey/distribution guidance.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int | None = None,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
        activation: str = "silu",
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = self.input_dim if output_dim is None else int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        activation_layer = _activation_factory(activation)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.hidden_dim),
            activation_layer(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        layers: list[nn.Module] = []
        in_dim = self.input_dim + self.hidden_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(activation_layer())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim

        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

        if zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape [batch, dim], got {tuple(x.shape)}.")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected x dimension {self.input_dim}, got {x.shape[1]}."
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
        h = torch.cat([x, t_emb], dim=-1)
        h = self.net(h)
        return self.output(h)


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    activations: dict[str, Callable[[], nn.Module]] = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(activations))
        raise ValueError(f"Unknown activation '{name}'. Expected one of: {valid}.") from exc
