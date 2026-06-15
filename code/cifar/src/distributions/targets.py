import torch
from omegaconf import DictConfig


class GaussianTargetMarginal:
    def __init__(self, mean: float, variance: float):
        if variance <= 0:
            raise ValueError("variance must be positive")
        self.mean = float(mean)
        self.variance = float(variance)
        self.std = self.variance ** 0.5
        self.distribution = torch.distributions.Normal(self.mean, self.std)

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)

    def log_prob(self, value: torch.Tensor):
        mean = torch.as_tensor(self.mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.std, device=value.device, dtype=value.dtype)
        return torch.distributions.Normal(mean, std).log_prob(value)


def build_target_marginal(target_cfg: DictConfig) -> GaussianTargetMarginal:
    target_type = str(target_cfg.type)
    if target_type != "gaussian":
        raise ValueError(f"Unsupported Jeffrey target type: {target_type}")
    return GaussianTargetMarginal(
        mean=float(target_cfg.mean),
        variance=float(target_cfg.variance),
    )


def estimate_gaussian_marginal(values: torch.Tensor) -> torch.distributions.Normal:
    values = values.detach().float().flatten()
    mean = values.mean()
    std = values.std(unbiased=values.numel() > 1).clamp_min(1e-6)
    return torch.distributions.Normal(mean, std)
