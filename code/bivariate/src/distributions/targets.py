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

#this function estimates p_\theta(y), i.e. the marginal of the updated dim (y by default in the config, can be changed), based on the model. Since, technically tds, should use this and not the analytic marginal.
def estimate_model_marginal(samples: torch.Tensor, updated_dim: int):
    y_samples = samples[:, updated_dim]
    # model mean
    estimated_mean = y_samples.mean()
    # model std
    estimated_std = y_samples.std(unbiased=True)
    return torch.distributions.Normal(estimated_mean, estimated_std)

