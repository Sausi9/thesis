from dataclasses import dataclass
import torch


@dataclass
class SDEConfig:
    type: str
    t_min: float
    t_max: float
    beta_min: float = 0.1
    beta_max: float = 20
    sigma_min: float | None = None
    sigma_max: float | None = None


class VPSDE:
    """Placeholder interface for future VP/VE/sub-VP implementations."""

    def __init__(self, config: SDEConfig):
        self.config = config

    def beta(self, t):
        return self.config.beta_min + t * (self.config.beta_max - self.config.beta_min)

    def beta_integral(self, t):
        return (
            self.config.beta_min * t
            + 0.5 * (self.config.beta_max - self.config.beta_min) * t**2
        )

    # returns the mean and var used in the closed form for p_0t(x(t) | x(0)), see Appendix B in Score SDE paper.
    def marginal_prob(self, t):
        mean_coeff = torch.exp(-0.5 * self.beta_integral(t))
        var_coeff = 1 - torch.exp(-self.beta_integral(t))
        return mean_coeff, var_coeff

    # drift coefficient of forward SDE
    def drift(self, x, t):
        return -0.5 * self.beta(t) * x

    # diffusion coefficient of forward/reverse SDE
    def diffusion(self, t):
        return torch.sqrt((self.beta(t)))

    def forward_noising(self, x0, t):
        # reparametrization formulation
        epsilon = torch.randn_like(x0)
        mean_coeff, var_coeff = self.marginal_prob(t)
        mean = mean_coeff[:, None, None, None] * x0
        std = torch.sqrt(var_coeff)[:, None, None, None]
        x_t = mean + std * epsilon
        return x_t, epsilon, std

