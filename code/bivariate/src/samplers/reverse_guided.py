import warnings

import torch
from tqdm.auto import tqdm


def resolve_guidance_scale(guidance_scale, guidance_coeff=None) -> float:
    if guidance_coeff is not None:
        warnings.warn(
            "guidance_coeff is deprecated; use guidance_scale instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        guidance_scale = guidance_coeff
    return float(guidance_scale)


class GuidedReverseSDESampler:
    def __init__(
        self,
        sde,
        target_marginal,
        original_marginal,
        data_dim,
        updated_dim,
        num_samples,
        guidance_scale=0.55,
        guidance_start=0.5,
        guidance_coeff=None,
    ):
        self.sde = sde
        self.target_marginal = target_marginal
        self.original_marginal = original_marginal
        self.data_dim = data_dim
        self.updated_dim = updated_dim
        self.num_samples = num_samples
        self.guidance_scale = resolve_guidance_scale(
            guidance_scale,
            guidance_coeff,
        )

        if guidance_start is None:
            raise ValueError("guidance_start must be set.")
        guidance_start = float(guidance_start)
        if not 0.0 <= guidance_start < 1.0:
            raise ValueError(f"guidance_start must be in [0, 1), got {guidance_start}.")
        self.guidance_start = guidance_start

    def log_potential_clean(self, x0: torch.Tensor):
        y = x0[:, self.updated_dim]
        return self.target_marginal.log_prob(y) - self.original_marginal.log_prob(y)

    def estimate_x0(self, x_t, t, score):
        alpha = torch.exp(-0.5 * self.sde.beta_integral(t))
        sigma2 = 1.0 - torch.exp(-self.sde.beta_integral(t))

        return (x_t + sigma2[:, None] * score) / alpha[:, None]

    # This function uses log(rho(updated_dim)) in the Jeffrey note context. Computes the score i.e. the gradient of it and adds it to the original score.
    def guided_score(self, model, x_t, t, guidance_scale):
        x_req = x_t.detach().requires_grad_(True)

        score = model(x_req, t)
        x0_hat = self.estimate_x0(x_req, t, score)
        log_potential = self.log_potential_clean(x0_hat)

        potential_grad = torch.autograd.grad(
            outputs=log_potential.sum(),
            inputs=x_req,
            create_graph=False,
        )[0]

        return (score + guidance_scale * potential_grad).detach()

    def guidance_scale_at_t(self, t):
        t_max = self.sde.config.t_max
        t_min = self.sde.config.t_min
        progress = ((t_max - t) / (t_max - t_min)).item()

        if progress < self.guidance_start:
            return 0.0
        return self.guidance_scale

    # Euler-Maruyama but with updated score for guidance
    def sample(self, model, num_steps, device, return_mean, progress):
        model.eval()
        x = self.sde.prior_sample((self.num_samples, self.data_dim), device)
        timesteps = torch.linspace(
            float(self.sde.config.t_max),
            float(self.sde.config.t_min),
            num_steps + 1,
            device=device,
        )

        iterator = range(num_steps)
        if progress:
            iterator = tqdm(iterator, desc="Sampling", leave=True)

        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            step_size = t - t_next
            t_batch = torch.full((self.num_samples,), t, device=device)

            # This score is basically the only difference from version in src/engine/sample.py
            guidance_scale = self.guidance_scale_at_t(t)
            score = self.guided_score(model, x, t_batch, guidance_scale)

            mean, variance = self.sde.reverse_transition_params(
                x,
                t_batch,
                score,
                step_size,
            )

            if i == num_steps - 1 and return_mean:
                x = mean
            else:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(variance)[:, None] * noise

        return x
