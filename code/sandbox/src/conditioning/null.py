from src.conditioning.base import ConditioningPotential
import torch


class NullConditioner(ConditioningPotential):
    def log_potential_x0(self, x0: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
