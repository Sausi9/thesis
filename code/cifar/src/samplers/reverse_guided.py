import torch
from tqdm.auto import tqdm

from src.diffusion.sde import batch_view
from src.jeffrey.brightness import brightness


class GuidedReverseSDESampler:
    def __init__(
        self,
        *,
        sde,
        target_marginal,
        original_marginal,
        num_samples,
        sample_shape,
        guidance_coeff=1.0,
        guidance_start=0.5,
        max_guidance_grad_norm=100.0,
    ):
        self.sde = sde
        self.target_marginal = target_marginal
        self.original_marginal = original_marginal
        self.num_samples = int(num_samples)
        self.sample_shape = tuple(int(v) for v in sample_shape)
        self.guidance_coeff = float(guidance_coeff)

        if guidance_start is None:
            raise ValueError("guidance_start must be set.")
        guidance_start = float(guidance_start)
        if not 0.0 <= guidance_start < 1.0:
            raise ValueError(f"guidance_start must be in [0, 1), got {guidance_start}.")
        self.guidance_start = guidance_start

        self.max_guidance_grad_norm = max_guidance_grad_norm
        if self.max_guidance_grad_norm is not None:
            self.max_guidance_grad_norm = float(self.max_guidance_grad_norm)
            if self.max_guidance_grad_norm <= 0.0:
                raise ValueError(
                    "max_guidance_grad_norm must be positive when it is set."
                )

    def estimate_x0(self, x_t, t, score):
        alpha = torch.exp(-0.5 * self.sde.beta_integral(t))
        sigma2 = 1.0 - torch.exp(-self.sde.beta_integral(t))
        return (x_t + batch_view(sigma2, x_t) * score) / batch_view(alpha, x_t)

    def log_potential(self, score, x_t, t):
        x0_hat = self.estimate_x0(x_t, t, score)
        y_hat = brightness(x0_hat)
        target_log_prob = self.target_marginal.log_prob(y_hat)
        original_log_prob = self._normal_log_prob(self.original_marginal, y_hat)
        return target_log_prob - original_log_prob

    @staticmethod
    def _normal_log_prob(marginal, value):
        loc = getattr(marginal, "loc", getattr(marginal, "mean", None))
        scale = getattr(marginal, "scale", getattr(marginal, "std", None))
        if loc is None or scale is None:
            raise TypeError("Expected a normal-like marginal with loc/scale or mean/std.")
        loc = torch.as_tensor(loc, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(scale, device=value.device, dtype=value.dtype)
        return torch.distributions.Normal(loc, scale).log_prob(value)

    def guidance_coeff_at_t(self, t):
        t_max = self.sde.config.t_max
        t_min = self.sde.config.t_min
        progress = (t_max - t) / (t_max - t_min)
        return torch.where(
            progress < self.guidance_start,
            torch.zeros_like(t),
            torch.full_like(t, self.guidance_coeff),
        )

    def guided_score(self, model, x_t, t):
        x_req = x_t.detach().requires_grad_(True)
        score = model(x_req, t)
        coeff = self.guidance_coeff_at_t(t).to(dtype=score.dtype)

        if torch.all(coeff == 0):
            return score.detach()

        log_potential = self.log_potential(score.detach(), x_req, t)
        potential_grad = torch.autograd.grad(log_potential.sum(), x_req)[0]

        if self.max_guidance_grad_norm is not None:
            potential_grad = torch.nan_to_num(
                potential_grad,
                nan=0.0,
                posinf=self.max_guidance_grad_norm,
                neginf=-self.max_guidance_grad_norm,
            )
            grad_norm = potential_grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            grad_scale = (
                self.max_guidance_grad_norm / grad_norm.clamp_min(1e-12)
            ).clamp(max=1.0)
            potential_grad = potential_grad * grad_scale

        guided = score + batch_view(coeff, score) * potential_grad
        return guided.detach()

    def sample(self, model, num_steps, device, return_mean=True, progress=True):
        model.eval()
        x = self.sde.prior_sample((self.num_samples, *self.sample_shape), device)
        timesteps = torch.linspace(
            float(self.sde.config.t_max),
            float(self.sde.config.t_min),
            int(num_steps) + 1,
            device=device,
        )

        iterator = range(int(num_steps))
        if progress:
            iterator = tqdm(iterator, desc="Naive guidance sampling", leave=True)

        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            step_size = t - t_next
            t_batch = torch.full((self.num_samples,), t, device=device)

            score = self.guided_score(model, x, t_batch)
            mean, variance = self.sde.reverse_transition_params(
                x,
                t_batch,
                score,
                step_size,
            )
            std = torch.sqrt(variance).view(self.num_samples, 1, 1, 1)

            if i == int(num_steps) - 1 and return_mean:
                x = mean
            else:
                x = mean + std * torch.randn_like(x)

        return x
