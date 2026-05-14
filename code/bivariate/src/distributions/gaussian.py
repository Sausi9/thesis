import torch


def extract_marginal_params(joint_mean, joint_covariance, dim: int):
    covariance = torch.as_tensor(joint_covariance, dtype=torch.float32)
    mean = torch.as_tensor(joint_mean, dtype=torch.float32)
    return mean[dim], covariance[dim, dim]


def calculate_conditional_params(joint_mean: list[float], joint_covariance: list[list[float]], marginal_dim: int, given_dim: int) -> dict:
    x_mean = joint_mean[marginal_dim]
    y_mean = joint_mean[given_dim]

    var_x = joint_covariance[marginal_dim][marginal_dim]
    cov_xy = joint_covariance[marginal_dim][given_dim]
    var_y = joint_covariance[given_dim][given_dim]

    slope = cov_xy / var_y
    intercept = x_mean - slope * y_mean
    conditional_var = var_x - cov_xy**2 / var_y

    return {
            "intercept": intercept,
            "slope":slope,
            "variance": conditional_var
            }

def conditional_distribution(given_value, params: dict):
    intercept = params.get("intercept")
    slope = params.get("slope")
    variance = params.get("variance")
    mean = intercept + slope * given_value
    return torch.distributions.Normal(mean, variance ** 0.5)

def estimate_model_gaussian_params(samples: torch.Tensor):
    mean = samples.mean(dim=0)
    covariance = torch.cov(samples.T)
    return mean, covariance
