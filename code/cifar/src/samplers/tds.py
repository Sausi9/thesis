import torch
from tqdm.auto import tqdm

from src.diffusion.sde import batch_view
from src.jeffrey.brightness import brightness


class TDSSampler:
    def __init__(
        self,
        num_particles,
        sde,
        num_samples,
        sample_shape,
        target_marginal,
        original_marginal,
        num_steps,
        twist_type="brightness",
        guidance_scale=1.0,
        resample_type=None,
        guidance_ramp=None,
        guidance_start=0.0,
        adaptive_resampling=False,
        ess_threshold=1.0,
        max_guidance_grad_norm=100.0,
    ):
        self.num_particles = int(num_particles)
        self.sde = sde
        self.num_samples = int(num_samples)
        self.sample_shape = tuple(int(v) for v in sample_shape)
        self.target_marginal = target_marginal
        self.original_marginal = original_marginal
        self.num_steps = int(num_steps)
        self.twist_type = str(twist_type)
        if self.twist_type != "brightness":
            raise ValueError(f"Unsupported twist type {self.twist_type}")
        self.guidance_scale = float(guidance_scale)

        self.resample_type = resample_type
        self.guidance_ramp = guidance_ramp
        if guidance_start is None:
            raise ValueError("guidance_start must be set.")
        guidance_start = float(guidance_start)
        if not 0.0 <= guidance_start < 1.0:
            raise ValueError(f"guidance_start must be in [0, 1), got {guidance_start}.")
        self.guidance_start = guidance_start

        self.adaptive_resampling = bool(adaptive_resampling)
        self.ess_threshold = float(ess_threshold)
        self.max_guidance_grad_norm = max_guidance_grad_norm
        if self.max_guidance_grad_norm is not None:
            self.max_guidance_grad_norm = float(self.max_guidance_grad_norm)
            if self.max_guidance_grad_norm <= 0.0:
                raise ValueError(
                    "max_guidance_grad_norm must be positive when it is set."
                )

    def init_particles(self, num_samples, device) -> torch.Tensor:
        return self.sde.prior_sample(
            (num_samples, self.num_particles, *self.sample_shape),
            device,
        )

    def init_weights(
        self,
        init_particles,
        init_t_batch_flat,
        model,
    ) -> torch.Tensor:
        B, K = init_particles.shape[:2]
        x_T_flat = init_particles.reshape(B * K, *self.sample_shape)

        with torch.no_grad():
            score = model(x_T_flat, init_t_batch_flat)
            log_twist_flat = self.log_twist(score, x_T_flat, init_t_batch_flat)

        return log_twist_flat.reshape(B, K).detach()

    def systematic_resample(self, particles, log_weights):
        B, K = particles.shape[:2]
        weights = torch.softmax(log_weights, dim=1)
        cdf = weights.cumsum(dim=1)
        cdf[:, -1] = 1.0

        u = torch.rand(B, 1, device=particles.device, dtype=weights.dtype) / K
        offsets = torch.arange(K, device=particles.device, dtype=weights.dtype) / K
        points = u + offsets[None, :]
        ancestor_idx = torch.searchsorted(cdf, points, right=False)

        batch_idx = torch.arange(B, device=particles.device)[:, None]
        resampled_particles = particles[batch_idx, ancestor_idx]
        return resampled_particles, torch.zeros_like(log_weights)

    def multinomial_resample(self, particles, log_weights):
        weights = torch.softmax(log_weights, dim=1)
        ancestor_idx = torch.multinomial(
            weights,
            num_samples=self.num_particles,
            replacement=True,
        )
        batch_idx = torch.arange(particles.shape[0], device=particles.device)[:, None]
        resampled_particles = particles[batch_idx, ancestor_idx]
        return resampled_particles, torch.zeros_like(log_weights)

    def resample(self, particles, log_weights):
        if self.resample_type == "multinomial":
            return self.multinomial_resample(particles, log_weights)
        if self.resample_type == "systematic":
            return self.systematic_resample(particles, log_weights)
        raise ValueError(f"Unsupported resample algorithm {self.resample_type}")

    def estimate_x0(self, x_t, t, score):
        alpha = torch.exp(-0.5 * self.sde.beta_integral(t))
        sigma2 = 1.0 - torch.exp(-self.sde.beta_integral(t))
        return (x_t + batch_view(sigma2, x_t) * score) / batch_view(alpha, x_t)

    def score_approx(self, model, x_t, t):
        x_req = x_t.detach().requires_grad_(True)
        score = model(x_req, t)
        log_twist = self.log_twist(score.detach(), x_req, t)

        self.check_tensor("score", score)
        self.check_tensor("log_twist_score_approx", log_twist)

        grad_log_twist = torch.autograd.grad(log_twist.sum(), x_req)[0]
        if self.max_guidance_grad_norm is not None:
            grad_log_twist = torch.nan_to_num(
                grad_log_twist,
                nan=0.0,
                posinf=self.max_guidance_grad_norm,
                neginf=-self.max_guidance_grad_norm,
            )
            grad_norm = grad_log_twist.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            grad_scale = (
                self.max_guidance_grad_norm / grad_norm.clamp_min(1e-12)
            ).clamp(max=1.0)
            grad_log_twist = grad_log_twist * grad_scale

        self.check_tensor("grad_log_twist", grad_log_twist)
        conditional_score_approx = (score + grad_log_twist).detach()
        self.check_tensor("conditional_score_approx", conditional_score_approx)
        return conditional_score_approx

    def proposal(self, model, x_prev, t_prev, step_size):
        conditional_score_approx = self.score_approx(model, x_prev, t_prev)
        proposal_mean, proposal_var = self.sde.reverse_transition_params(
            x_prev,
            t_prev,
            conditional_score_approx,
            step_size,
        )
        self.check_tensor("proposal_mean", proposal_mean)
        self.check_tensor("proposal_var", proposal_var)

        proposal_std = torch.sqrt(proposal_var).view(-1, 1, 1, 1)
        return (proposal_mean + proposal_std * torch.randn_like(proposal_mean)).detach()

    def guidance_linear(self, t):
        t_max = self.sde.config.t_max
        t_min = self.sde.config.t_min
        lambda_linear = (t_max - t) / (t_max - t_min)
        return lambda_linear.clamp(0.0, 1.0)

    def guidance_delayed_linear(self, t):
        t_max = self.sde.config.t_max
        t_min = self.sde.config.t_min
        progress = (t_max - t) / (t_max - t_min)
        lambda_delayed_linear = (progress - self.guidance_start) / (
            1.0 - self.guidance_start
        )
        return lambda_delayed_linear.clamp(0.0, 1.0)

    def guidance_delayed_discrete(self, t):
        t_max = self.sde.config.t_max
        t_min = self.sde.config.t_min
        progress = (t_max - t) / (t_max - t_min)
        return torch.where(
            progress < self.guidance_start,
            torch.zeros_like(t),
            torch.ones_like(t),
        )

    def guidance_strength(self, t):
        if self.guidance_ramp is None:
            return torch.ones_like(t)
        if self.guidance_ramp not in ("linear", "delayed_linear", "delayed_discrete"):
            raise ValueError(f"Unsupported guidance ramp {self.guidance_ramp}")
        if self.guidance_ramp == "linear":
            return self.guidance_linear(t)
        if self.guidance_ramp == "delayed_linear":
            return self.guidance_delayed_linear(t)
        return self.guidance_delayed_discrete(t)

    def log_twist(self, score, x_t, t):
        x0_hat = self.estimate_x0(x_t, t, score)
        y_hat = brightness(x0_hat)
        target_log_prob = self.target_marginal.log_prob(y_hat)
        original_log_prob = self._normal_log_prob(self.original_marginal, y_hat)
        base_log_twist = target_log_prob - original_log_prob
        strength = self.guidance_strength(t).to(dtype=base_log_twist.dtype)
        return self.guidance_scale * strength * base_log_twist

    @staticmethod
    def _normal_log_prob(marginal, value):
        loc = getattr(marginal, "loc", getattr(marginal, "mean", None))
        scale = getattr(marginal, "scale", getattr(marginal, "std", None))
        if loc is None or scale is None:
            raise TypeError("Expected a normal-like marginal with loc/scale or mean/std.")
        loc = torch.as_tensor(loc, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(scale, device=value.device, dtype=value.dtype)
        return torch.distributions.Normal(loc, scale).log_prob(value)

    def check_tensor(self, name, x):
        finite = torch.isfinite(x)
        if not finite.all():
            num_nan = torch.isnan(x).sum().item()
            num_posinf = torch.isposinf(x).sum().item()
            num_neginf = torch.isneginf(x).sum().item()
            finite_x = x[finite]
            finite_min = finite_x.min().item() if finite_x.numel() else None
            finite_max = finite_x.max().item() if finite_x.numel() else None
            raise FloatingPointError(
                f"{name} nonfinite: "
                f"nan={num_nan}, +inf={num_posinf}, -inf={num_neginf}, "
                f"finite_min={finite_min}, finite_max={finite_max}"
            )

    def log_weight(self, model, x_t, x_prev, t_current, t_prev, step_size):
        with torch.no_grad():
            score_prev = model(x_prev, t_prev)
            score_current = model(x_t, t_current)
            transition_mean_prev, transition_var_prev = (
                self.sde.reverse_transition_params(
                    x_prev,
                    t_prev,
                    score_prev,
                    step_size,
                )
            )

        conditional_score_approx = self.score_approx(model, x_prev, t_prev)
        with torch.no_grad():
            mean_q, _ = self.sde.reverse_transition_params(
                x_prev,
                t_prev,
                conditional_score_approx,
                step_size,
            )

            mean_diff = (mean_q - transition_mean_prev).double()
            proposal_residual = (x_t - mean_q).double()
            transition_var_prev = transition_var_prev.double()

            reduce_dims = tuple(range(1, x_t.ndim))
            cross = (proposal_residual * mean_diff).sum(dim=reduce_dims)
            norm = mean_diff.square().sum(dim=reduce_dims)
            transition_log_ratio = -0.5 * (2.0 * cross + norm) / transition_var_prev

            log_twist_current = self.log_twist(score_current, x_t, t_current)
            log_twist_prev = self.log_twist(score_prev, x_prev, t_prev)

            self.check_tensor("transition_log_ratio", transition_log_ratio)
            self.check_tensor("log_twist_current", log_twist_current)
            self.check_tensor("log_twist_prev", log_twist_prev)

            log_weight = transition_log_ratio + log_twist_current - log_twist_prev
            self.check_tensor("log_weight", log_weight)

        return log_weight.detach()

    def ess(self, log_weights):
        weights = torch.softmax(log_weights, dim=1)
        squared_weights = weights.pow(2)
        return 1.0 / squared_weights.sum(dim=1)

    def sample(self, model, device, progress=True):
        K = self.num_particles
        B = self.num_samples
        particles = self.init_particles(B, device)

        init_t_batch_flat = torch.full(
            (B * K,),
            self.sde.config.t_max,
            device=device,
        )
        log_weights = self.init_weights(particles, init_t_batch_flat, model)
        log_norm = torch.logsumexp(log_weights, dim=1, keepdim=True)
        if not torch.isfinite(log_norm).all():
            raise FloatingPointError("Non-finite initial log norm")
        log_weights = log_weights - log_norm

        timesteps = torch.linspace(
            self.sde.config.t_max,
            self.sde.config.t_min,
            self.num_steps + 1,
            device=device,
        )

        iterator = range(self.num_steps)
        if progress:
            iterator = tqdm(iterator, desc="TDS sampling", leave=True)

        for i in iterator:
            t_prev = timesteps[i]
            t = timesteps[i + 1]
            step_size = t_prev - t

            t_prev_flat = torch.full((B * K,), t_prev.item(), device=device)
            t_flat = torch.full((B * K,), t.item(), device=device)

            with torch.no_grad():
                mean_ess = self.ess(log_weights).mean()
                if self.adaptive_resampling and mean_ess < self.ess_threshold * K:
                    particles, log_weights = self.resample(particles, log_weights)
                if not self.adaptive_resampling:
                    particles, log_weights = self.resample(particles, log_weights)
                old_log_weights = log_weights.detach()

            x_prev_flat = particles.reshape(B * K, *self.sample_shape)
            x_flat = self.proposal(model, x_prev_flat, t_prev_flat, step_size)
            log_incremental_weights_flat = self.log_weight(
                model,
                x_flat,
                x_prev_flat,
                t_flat,
                t_prev_flat,
                step_size,
            ).reshape(B, K)

            particles = x_flat.reshape(B, K, *self.sample_shape).detach()
            log_weights = (old_log_weights + log_incremental_weights_flat).detach()
            log_norm = torch.logsumexp(log_weights, dim=1, keepdim=True)
            if not torch.isfinite(log_norm).all():
                raise FloatingPointError("Non-finite log norm")
            log_weights = log_weights - log_norm

        weights = torch.softmax(log_weights, dim=1)
        chosen_idx = torch.multinomial(weights, 1).squeeze(1)
        batch_idx = torch.arange(particles.shape[0], device=device)
        return particles[batch_idx, chosen_idx]
