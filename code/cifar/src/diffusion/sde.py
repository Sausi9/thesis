from dataclasses import dataclass

import torch


@dataclass
class SDEConfig:
    type: str
    t_min: float
    t_max: float
    beta_min: float = 0.1
    beta_max: float = 20.0
    sigma_min: float | None = None
    sigma_max: float | None = None


def batch_view(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return value.view(value.shape[0], *([1] * (like.ndim - 1)))


class VPSDE:
    def __init__(self, config: SDEConfig):
        self.config = config

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.config.beta_min + t * (self.config.beta_max - self.config.beta_min)

    def beta_integral(self, t: torch.Tensor) -> torch.Tensor:
        return (
            self.config.beta_min * t
            + 0.5 * (self.config.beta_max - self.config.beta_min) * t**2
        )

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
        mean_coeff = torch.exp(-0.5 * self.beta_integral(t))
        var_coeff = 1.0 - torch.exp(-self.beta_integral(t))
        mean = batch_view(mean_coeff, x0) * x0
        std = batch_view(torch.sqrt(var_coeff), x0)
        return mean, std

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * batch_view(self.beta(t), x) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta(t))

    def sde(self, x: torch.Tensor, t: torch.Tensor):
        drift = self.drift(x, t)
        diffusion = self.diffusion(t)
        return drift, diffusion

    def forward_noising(self, x0: torch.Tensor, t: torch.Tensor):
        epsilon = torch.randn_like(x0)
        mean, std = self.marginal_prob(x0, t)
        x_t = mean + std * epsilon
        return x_t, epsilon, std

    def prior_sample(self, shape, device):
        return torch.randn(shape, device=device)

    def reverse_drift(self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor):
        drift, diffusion = self.sde(x, t)
        return drift - batch_view(diffusion.square(), x) * score

    def reverse_transition_params(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        step_size: torch.Tensor | float,
    ):
        reverse_drift = self.reverse_drift(x, t, score)
        mean = x - reverse_drift * step_size
        variance = self.beta(t) * step_size
        return mean, variance
