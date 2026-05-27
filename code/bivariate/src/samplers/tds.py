import torch
from tqdm.auto import tqdm


class TDSSampler:
    def __init__(
        self,
        num_particles,
        sde,
        num_samples,
        target_marginal,
        original_marginal,
        updated_dim,
        data_dim,
        num_steps,
        twist_type="tractable",
        base_mean=None,
        base_covariance=None,
        updated_mean=None,
        updated_covariance=None,
        resample_type=None,
        guidance_ramp=None,
        guidance_start=0.0,
        adaptive_resampling=False,
        ess_threshold=1.0,
    ):
        self.num_particles = num_particles
        self.sde = sde
        self.num_samples = num_samples
        self.target_marginal = target_marginal
        self.original_marginal = original_marginal
        self.updated_dim = updated_dim
        self.data_dim = data_dim
        self.num_steps = num_steps

        self.twist_type = twist_type

        self.base_mean = (
            None
            if base_mean is None
            else torch.as_tensor(base_mean, dtype=torch.float32)
        )
        self.base_covariance = (
            None
            if base_covariance is None
            else torch.as_tensor(base_covariance, dtype=torch.float32)
        )
        self.updated_mean = (
            None
            if updated_mean is None
            else torch.as_tensor(updated_mean, dtype=torch.float32)
        )
        self.updated_covariance = (
            None
            if updated_covariance is None
            else torch.as_tensor(updated_covariance, dtype=torch.float32)
        )

        if self.twist_type == "optimal":
            missing = [
                name
                for name, value in {
                    "base_mean": self.base_mean,
                    "base_covariance": self.base_covariance,
                    "updated_mean": self.updated_mean,
                    "updated_covariance": self.updated_covariance,
                }.items()
                if value is None
            ]
            if missing:
                raise ValueError(f"Optimal twist requires: {', '.join(missing)}")
        self.resample_type = resample_type
        self.guidance_ramp = guidance_ramp
        if guidance_start is None:
            raise ValueError("guidance_start must be set.")

        guidance_start = float(guidance_start)
        if not 0.0 <= guidance_start < 1.0:
            raise ValueError(f"guidance_start must be in [0, 1), got {guidance_start}.")

        self.guidance_start = guidance_start
        self.adaptive_resampling = adaptive_resampling
        self.ess_threshold = ess_threshold

    def init_particles(self, K, num_samples, sample_shape, device) -> torch.Tensor:
        # x^T ~ p(x^T), where p(x^T) is standard normal
        particles = self.sde.prior_sample(
            (num_samples, self.num_particles, self.data_dim), device
        )
        return particles

    def init_weights(
        self,
        init_particles,
        init_t_batch_flat,
        model,
        device,
    ) -> torch.Tensor:
        B, K, D = init_particles.shape

        # init particles, flattened.
        x_T_flat = init_particles.reshape(B * K, D)

        with torch.no_grad():
            score = model(x_T_flat, init_t_batch_flat)
            log_twist_flat = self.log_twist(score, x_T_flat, init_t_batch_flat)

        log_weights = log_twist_flat.reshape(B, K).detach()
        return log_weights

    def systematic_resample(self, particles, log_weights):
        B, K, D = particles.shape

        weights = torch.softmax(log_weights, dim=1)  # [B, K]
        cdf = weights.cumsum(dim=1)  # [B, K]
        cdf[:, -1] = 1.0

        u = torch.rand(B, 1, device=particles.device, dtype=weights.dtype) / K

        # K evenly spaced points per batch item.
        offsets = torch.arange(K, device=particles.device, dtype=weights.dtype) / K
        points = u + offsets[None, :]  # [B, K]

        # For each point, find which CDF bin contains it.
        ancestor_idx = torch.searchsorted(cdf, points, right=False)

        batch_idx = torch.arange(B, device=particles.device)[:, None]
        resampled_particles = particles[batch_idx, ancestor_idx]  # [B, K, D]

        return resampled_particles, torch.zeros_like(log_weights)

    def multinomial_resample(self, particles, log_weights):
        weights = torch.softmax(log_weights, dim=1)  # [B, K]
        ancestor_idx = torch.multinomial(
            weights,
            num_samples=self.num_particles,
            replacement=True,
        )  # [B, K]

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

        return (x_t + sigma2[:, None] * score) / alpha[:, None]

    def score_approx(self, model, x_t, t):
        # This is to be able to compute grads.
        x_req = x_t.detach().requires_grad_(True)

        # The paper reconstructs the score from the denoising estimate of x_0, we have it directly from model.
        score = model(x_req, t)

        log_twist = self.log_twist(score, x_req, t)

        grad_log_twist = torch.autograd.grad(log_twist.sum(), x_req)[0]

        conditional_score_approx = (score + grad_log_twist).detach()
        return conditional_score_approx

    # returns proposal which is a x^t_k
    def proposal(self, model, x_prev, t_prev, step_size):
        conditional_score_approx = self.score_approx(model, x_prev, t_prev)

        proposal_mean, proposal_var = self.sde.reverse_transition_params(
            x_prev,
            t_prev,
            conditional_score_approx,
            step_size,
        )

        proposal_std = torch.sqrt(proposal_var)[:, None]

        x_t = torch.distributions.Normal(proposal_mean, proposal_std).sample().detach()
        return x_t

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
            1 - self.guidance_start
        )

        return lambda_delayed_linear.clamp(0.0, 1.0)

    def guidance_strength(self, t):
        if self.guidance_ramp is None:
            return torch.ones_like(t)
        if self.guidance_ramp != "linear" and self.guidance_ramp != "delayed_linear":
            raise ValueError(f"Unsupported guidance ramp {self.guidance_ramp}")

        if self.guidance_ramp == "linear":
            return self.guidance_linear(t)
        if self.guidance_ramp == "delayed_linear":
            return self.guidance_delayed_linear(t)

    def log_twist(self, score, x_t, t):
        if self.twist_type == "tractable":
            return self.log_tractable_twist(score, x_t, t)
        if self.twist_type == "optimal":
            return self.log_optimal_twist(x_t, t)
        raise ValueError(f"Unsupported twist type {self.twist_type}")

    # returns log of p^*(y)/p(y) i.e. rho(y) in Jeffrey note. Using log allows for subtracting instead of dividing.
    def log_tractable_twist(self, score, x_t, t):
        x0_hat = self.estimate_x0(x_t, t, score)
        y_hat = x0_hat[:, self.updated_dim]

        mean_target = torch.as_tensor(
            self.target_marginal.mean,
            device=y_hat.device,
            dtype=y_hat.dtype,
        )
        var_target = torch.as_tensor(
            self.target_marginal.variance,
            device=y_hat.device,
            dtype=y_hat.dtype,
        )

        mean_original = torch.as_tensor(
            self.original_marginal.mean,
            device=y_hat.device,
            dtype=y_hat.dtype,
        )
        var_original = torch.as_tensor(
            self.original_marginal.variance,
            device=y_hat.device,
            dtype=y_hat.dtype,
        )

        # base_log_twist = self.target_marginal.log_prob(y_hat) - self.original_marginal.log_prob(
        #     y_hat
        # )
        a = 1.0 / var_target - 1.0 / var_original
        b = -2.0 * mean_target / var_target + 2.0 * mean_original / var_original
        c = mean_target.square() / var_target - mean_original.square() / var_original

        base_log_twist = -0.5 * (
            a * y_hat.square() + b * y_hat + c + torch.log(var_target / var_original)
        )
        return self.guidance_strength(t) * base_log_twist

    # this function uses the exact/optimal twist, analogous to the optimal twist in TDS paper. It is not generally tractable, however in this bivariate toy example it is. Used as a baseline to compare the differences between the twists.
    def log_optimal_twist(self, x_t, t_batch):
        assert self.base_mean is not None
        assert self.base_covariance is not None
        assert self.updated_mean is not None
        assert self.updated_covariance is not None

        base_mean = self.base_mean.to(device=x_t.device, dtype=x_t.dtype)
        base_cov = self.base_covariance.to(device=x_t.device, dtype=x_t.dtype)
        updated_mean = self.updated_mean.to(device=x_t.device, dtype=x_t.dtype)
        updated_cov = self.updated_covariance.to(device=x_t.device, dtype=x_t.dtype)

        t_scalar = t_batch[0]

        beta_int = self.sde.beta_integral(t_scalar)
        alpha = torch.exp(-0.5 * beta_int)
        sigma2 = 1.0 - torch.exp(-beta_int)

        eye = torch.eye(x_t.shape[1], device=x_t.device, dtype=x_t.dtype)

        base_mean_t = alpha * base_mean
        updated_mean_t = alpha * updated_mean

        base_cov_t = alpha**2 * base_cov + sigma2 * eye
        updated_cov_t = alpha**2 * updated_cov + sigma2 * eye

        base_dist = torch.distributions.MultivariateNormal(
            base_mean_t,
            covariance_matrix=base_cov_t,
        )
        updated_dist = torch.distributions.MultivariateNormal(
            updated_mean_t,
            covariance_matrix=updated_cov_t,
        )

        return updated_dist.log_prob(x_t) - base_dist.log_prob(x_t)

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
                    x_prev, t_prev, score_prev, step_size
                )
            )

        conditional_score_approx = self.score_approx(model, x_prev, t_prev)
        with torch.no_grad():
            mean_q, _ = self.sde.reverse_transition_params(
                x_prev, t_prev, conditional_score_approx, step_size
            )

            mean_diff = mean_q - transition_mean_prev
            proposal_residual = x_t - mean_q
            # Direct computation of log p(x_t | x_prev) - log q(x_t | x_prev).
            # Both transitions have the same variance in the VP SDE.
            transition_log_ratio = -0.5 * (
                2.0 * (proposal_residual * mean_diff).sum(dim=1)
                + mean_diff.square().sum(dim=1)
            ) / transition_var_prev

            # current and prev log twists, that is \tilde{p} for t and t+1
            log_twist_current = self.log_twist(score_current, x_t, t_current)
            log_twist_prev = self.log_twist(score_prev, x_prev, t_prev)

            self.check_tensor("transition_log_ratio", transition_log_ratio)
            self.check_tensor("log_twist_current", log_twist_current)
            self.check_tensor("log_twist_prev", log_twist_prev)

            log_weight = (
                transition_log_ratio
                + log_twist_current
                - log_twist_prev
            )

            self.check_tensor("log_weight", log_weight)

        return log_weight.detach()

    def ess(self, log_weights):
        weights = torch.softmax(log_weights, dim=1)  # [B, K]
        squared_weights = weights.pow(2)
        ess = 1.0 / (squared_weights.sum(dim=1))  # [B]
        return ess

    def sample(self, model, device, progress=True):
        K = self.num_particles
        B = self.num_samples
        D = self.data_dim
        # [B, K, D]
        particles = self.init_particles(K, B, D, device)

        init_t_batch_flat = torch.full(
            (B * K,),
            self.sde.config.t_max,
            device=device,
        )

        # [B, K]
        log_weights = self.init_weights(
            particles,
            init_t_batch_flat,
            model,
            device,
        )

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
            # t+1
            t_prev = timesteps[i]
            # t
            t = timesteps[i + 1]

            step_size = t_prev - t

            t_prev_flat = torch.full((B * K,), t_prev.item(), device=device)
            t_flat = torch.full((B * K,), t.item(), device=device)

            with torch.no_grad():
                mean_ess = self.ess(log_weights).mean()
                if self.adaptive_resampling and mean_ess < self.ess_threshold * K:
                    # log_weights are returned as 0 here.
                    particles, log_weights = self.resample(particles, log_weights)

                if not self.adaptive_resampling:
                    particles, log_weights = self.resample(particles, log_weights)

                # store the old log_weights or the reset (zero) log_weights in the case where we resample in old_log_weights variable
                old_log_weights = log_weights.detach()

            x_prev_flat = particles.reshape(B * K, D)

            # propose all particles at once
            # conditional score approx is called within proposal
            x_flat = self.proposal(model, x_prev_flat, t_prev_flat, step_size)

            # compute all weights at once
            # twisting function is called within log_weight
            log_incremental_weights_flat = self.log_weight(
                model, x_flat, x_prev_flat, t_flat, t_prev_flat, step_size
            ).reshape(B, K)

            particles = x_flat.reshape(B, K, D).detach()
            # this accumulates weights, by adding old_log_weights and the newly computed weights. If we did not resample, then
            log_weights = (old_log_weights + log_incremental_weights_flat).detach()
            log_norm = torch.logsumexp(log_weights, dim=1, keepdim=True)
            if not torch.isfinite(log_norm).all():
                raise FloatingPointError("Non-finite log norm")
            log_weights = log_weights - log_norm

        # particles: [B, K, D]
        # log_weights: [B, K]

        weights = torch.softmax(log_weights, dim=1)  # [B, K]
        chosen_idx = torch.multinomial(weights, 1).squeeze(1)  # [B]

        batch_idx = torch.arange(particles.shape[0], device=device)
        return particles[batch_idx, chosen_idx]
