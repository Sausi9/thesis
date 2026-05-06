import torch
from omegaconf import DictConfig

# new Gaussian marginal for y, its a univariate distribution for this toy example
class GaussianTargetMarginal:
    def __init__(self, mean: float, variance: float):
        if variance <= 0:
            raise ValueError("variance must be positive")
        self.mean = mean
        self.variance = variance
        self.std = variance ** 0.5
        self.distribution = torch.distributions.Normal(self.mean, self.std)

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)

    def log_prob(self, value):
        return self.distribution.log_prob(value)


def build_target_marginal(target_cfg: DictConfig) -> GaussianTargetMarginal:
    target_type = str(target_cfg.type)
    if target_type != "gaussian":
        raise ValueError(f"Unsupported Jeffrey target type: {target_type}")

    return GaussianTargetMarginal(
        mean=float(target_cfg.mean),
        variance=float(target_cfg.variance),
    )
