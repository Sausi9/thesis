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
    def marginal_prob(self, x0, t):
        mean_coeff = torch.exp(-0.5 * self.beta_integral(t))
        var_coeff = 1 - torch.exp(-self.beta_integral(t))
        mean = mean_coeff[:, None] * x0
        std = torch.sqrt(var_coeff)[:, None]
        return mean, std

    # drift coefficient of forward SDE
    def drift(self, x, t):
        return -0.5 * self.beta(t)[:, None] * x

    # diffusion coefficient of forward/reverse SDE
    def diffusion(self, t):
        return torch.sqrt((self.beta(t)))
    
    # just a method that returns both drift and diffusion coefficients, official score sde does this. Common. For convenience
    def sde(self, x, t):
        drift = self.drift(x,t)
        diffusion = self.diffusion(t)
        return drift, diffusion

    def forward_noising(self, x0, t):
        # reparametrization formulation
        epsilon = torch.randn_like(x0)
        mean, std = self.marginal_prob(x0, t)
        x_t = mean + std * epsilon
        return x_t, epsilon, std
    
    def prior_sample(self, shape, device):
        return torch.randn(shape, device = device)

    def reverse_drift(self, x, t, score):
        drift, diffusion = self.sde(x, t)
        return drift - diffusion[:, None] ** 2 * score

