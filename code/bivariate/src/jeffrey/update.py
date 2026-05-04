import torch

from src.distributions.gaussian import calculate_conditional_params


# this function calculates the mean and cov of the new joint (after the new marginal is used)
# this is possible here, because the new joint is also Gaussian
def jeffrey_updated_gaussian_params(
    joint_mean,
    joint_covariance,
    updated_dim,
    target_marginal,
):
    mean = torch.as_tensor(joint_mean, dtype=torch.float32)
    covariance = torch.as_tensor(joint_covariance, dtype=torch.float32)

    if mean.shape != (2,):
        raise ValueError(f"joint_mean must have shape (2,), got {tuple(mean.shape)}")
    if covariance.shape != (2, 2):
        raise ValueError(
            f"joint_covariance must have shape (2, 2), got {tuple(covariance.shape)}"
        )
    if updated_dim not in (0, 1):
        raise ValueError("updated_dim must be 0 or 1 for this bivariate example")

    kept_dim = 1 - updated_dim

    conditional = calculate_conditional_params(
        mean.tolist(),
        covariance.tolist(),
        marginal_dim=kept_dim,
        given_dim=updated_dim,
    )

    slope = conditional["slope"]
    intercept = conditional["intercept"]
    conditional_var = conditional["variance"]

    target_mean = target_marginal.mean
    target_var = target_marginal.variance

    updated_mean = mean.clone()
    updated_covariance = covariance.clone()

    updated_mean[updated_dim] = target_mean
    updated_mean[kept_dim] = intercept + slope * target_mean

    updated_covariance[updated_dim, updated_dim] = target_var
    updated_covariance[kept_dim, updated_dim] = slope * target_var
    updated_covariance[updated_dim, kept_dim] = slope * target_var
    updated_covariance[kept_dim, kept_dim] = conditional_var + slope**2 * target_var

    return updated_mean, updated_covariance


# This function uses the factorization of the joint into a conditional and marginal
# and then samples from the marginal and then the conditional, which mirrors the Jeffrey's update
# directly. 
def sample_jeffrey_update(
    joint_mean,
    joint_covariance,
    updated_dim,
    target_marginal,
    num_samples,
):
    mean = torch.as_tensor(joint_mean, dtype=torch.float32)
    covariance = torch.as_tensor(joint_covariance, dtype=torch.float32)

    if updated_dim not in (0, 1):
        raise ValueError("updated_dim must be 0 or 1 for this bivariate example")

    kept_dim = 1 - updated_dim
    conditional = calculate_conditional_params(
        mean.tolist(),
        covariance.tolist(),
        marginal_dim=kept_dim,
        given_dim=updated_dim,
    )

    updated_values = target_marginal.sample((int(num_samples),))
    kept_mean = conditional["intercept"] + conditional["slope"] * updated_values
    kept_std = conditional["variance"] ** 0.5
    kept_values = kept_mean + kept_std * torch.randn_like(kept_mean)

    samples = torch.empty(int(num_samples), 2, dtype=torch.float32)
    samples[:, updated_dim] = updated_values
    samples[:, kept_dim] = kept_values
    return samples
