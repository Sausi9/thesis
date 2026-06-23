from __future__ import annotations

import torch
from torch import nn


class LinearRatioClassifier(nn.Module):
    """Linear binary classifier whose logit estimates a balanced density ratio."""

    def __init__(self, embedding_dim: int = 2048):
        super().__init__()
        self.linear = nn.Linear(int(embedding_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def standardize(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean) / std.clamp_min(1e-6)
