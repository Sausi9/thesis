from dataclasses import dataclass


@dataclass
class SDEConfig:
    type: str
    t_min: float
    t_max: float
    beta_min: float | None = None
    beta_max: float | None = None
    sigma_min: float | None = None
    sigma_max: float | None = None


class BaseSDE:
    """Placeholder interface for future VP/VE/sub-VP implementations."""

    def __init__(self, config: SDEConfig):
        self.config = config

    def marginal_prob(self, x, t):
        raise NotImplementedError("Implement marginal_prob for the chosen SDE.")

    def drift(self, x, t):
        raise NotImplementedError("Implement drift for the chosen SDE.")

    def diffusion(self, t):
        raise NotImplementedError("Implement diffusion for the chosen SDE.")
