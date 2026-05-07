import torch
import math
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
    ):
        self.num_particles = num_particles
        self.sde = sde
        self.num_samples = num_samples
        self.target_marginal = target_marginal
        self.original_marginal = original_marginal
        self.updated_dim = updated_dim
        self.data_dim = data_dim
        self.num_steps = num_steps

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

        score = model(x_T_flat, init_t_batch_flat)
        log_twist_flat = self.log_twist(score, x_T_flat, init_t_batch_flat)

        log_weights = log_twist_flat.reshape(B, K)
        return log_weights

    def resample(self, particles, log_weights):
        weights = torch.softmax(log_weights, dim=1)  # [B, K]

        ancestor_idx = torch.multinomial(
            weights,
            num_samples=self.num_particles,
            replacement=True,
        )  # [B, K]

        batch_idx = torch.arange(particles.shape[0], device=particles.device)[:, None]

        resampled_particles = particles[batch_idx, ancestor_idx]

        # TODO: THIS SHOULD BE REMOVED I THINK, IT RESETS THE WEIGHTS TO UNIFORM AGAIN, WAS LEFT OVER FROM OLD IMPLEMENTATION
        log_weights = torch.full(
            (particles.shape[0], self.num_particles),
            -math.log(self.num_particles),
            device=particles.device,
            dtype=particles.dtype,
        )

        return resampled_particles, log_weights

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

        conditional_score_approx = score + grad_log_twist
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

        x_t = torch.distributions.Normal(proposal_mean, proposal_std).sample()
        return x_t

    # returns [B, 1]
    def log_twist(self, score, x_t, t):
        x0_hat = self.estimate_x0(x_t, t, score)
        y_hat = x0_hat[:, self.updated_dim]

        return self.target_marginal.log_prob(y_hat) - self.original_marginal.log_prob(
            y_hat
        )

    def log_weight(self, model, x_t, x_prev, t_current, t_prev, step_size):
        score_prev = model(x_prev, t_prev)
        score_current = model(x_t, t_current)

        transition_mean_prev, transition_var_prev = self.sde.reverse_transition_params(
            x_prev, t_prev, score_prev, step_size
        )
        transition_dist_prev = torch.distributions.Normal(
            transition_mean_prev, torch.sqrt(transition_var_prev)[:, None]
        )
        # normal reverse log transition
        log_transition = transition_dist_prev.log_prob(x_t).sum(dim=1)

        conditional_score_approx = self.score_approx(model, x_prev, t_prev)
        mean_q, var_q = self.sde.reverse_transition_params(
            x_prev, t_prev, conditional_score_approx, step_size
        )
        twisted_transition_dist = torch.distributions.Normal(
            mean_q, torch.sqrt(var_q)[:, None]
        )
        # log of the twisted reverse transition, which uses the conditional score approx, see Eq.9 in TDS paper.
        log_twisted_transition = twisted_transition_dist.log_prob(x_t).sum(dim=1)

        # current and prev log twists, that is \tilde{p} for t and t+1
        log_twist_current = self.log_twist(score_current, x_t, t_current)
        log_twist_prev = self.log_twist(score_prev, x_prev, t_prev)

        log_weight = (
            log_transition + log_twist_current - log_twisted_transition - log_twist_prev
        )
        return log_weight

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

            particles, log_weights = self.resample(particles, log_weights)
            x_prev_flat = particles.reshape(B * K, D)

            # propose all particles at once
            # conditional score approx is called within proposal
            x_flat = self.proposal(model, x_prev_flat, t_prev_flat, step_size)

            # compute all weights at once
            # twisting function is called within log_weight
            log_w_flat = self.log_weight(
                model, x_flat, x_prev_flat, t_flat, t_prev_flat, step_size
            )

            particles = x_flat.reshape(B, K, D)
            log_weights = log_w_flat.reshape(B, K)

        # particles: [B, K, D]
        # log_weights: [B, K]

        weights = torch.softmax(log_weights, dim=1)  # [B, K]
        chosen_idx = torch.multinomial(weights, 1).squeeze(1)  # [B]

        batch_idx = torch.arange(particles.shape[0], device=device)
        return particles[batch_idx, chosen_idx]
