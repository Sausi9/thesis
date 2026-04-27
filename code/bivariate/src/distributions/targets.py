import torch

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
