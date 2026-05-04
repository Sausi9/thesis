import torch
import math


class TDSSampler:
    def __init__(
        self, num_particles, sde, conditioner, num_samples, sample_shape
    ):
        self.num_particles = num_particles
        self.diffusion = diffusion
        self.conditioner = conditioner
        self.num_samples = num_samples
        self.sample_shape = sample_shape

    def init_particles(
        self, K, num_samples, sample_shape, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        particles = torch.randn(
            num_samples,
            self.num_particles,
            *sample_shape,
            device=device,
        )
        # TODO: Implement weight initialization based on the conditioning distribution instead of uniform log weights, see paper.
        log_weights = torch.full(
            (num_samples, self.num_particles),
            -math.log(self.num_particles),
            device=device,
            dtype=particles.dtype,
        )
        return particles, log_weights

    def resample(self, particles, log_weights):
        weights = torch.softmax(log_weights, dim=1)  # [B, K]

        ancestor_idx = torch.multinomial(
            weights,
            num_samples=self.num_particles,
            replacement=True,
        )  # [B, K]

        batch_idx = torch.arange(particles.shape[0], device=particles.device)[:, None]

        resampled_particles = particles[batch_idx, ancestor_idx]

        log_weights = torch.full(
            (particles.shape[0], self.num_particles),
            -math.log(self.num_particles),
            device=particles.device,
            dtype=particles.dtype,
        )

        return resampled_particles, log_weights

    # def score_approx(self):
    # TODO: Add the conditional score approx to shift the mean and var from the sample, might need to refactor the sample_reverse_step function slightly, or have some way to shift the normal dist.
    def proposal(self, model, x_t, t):
        x_prev, _ = self.diffusion.sample_reverse_step(model, x_t, t)
        return x_prev

    def twist(self, model, x_t, t):
        return self.conditioner.log_potential_xt(self.diffusion, model, x_t, t)

    # TODO: Implement the actual weight formula, this is a simple replacement for now, just weighting with 0s everywhere
    def weight(self, model, x_t, x_prev, t):
        log_twist_t = self.twist(model, x_t, t)
        return torch.zeros_like(log_twist_t)

    def sample(self, model, device):
        B = self.num_samples
        K = self.num_particles
        sample_shape = self.sample_shape
        # [B, K]
        particles, log_weights = self.init_particles(K, B, sample_shape, device)
        for t in range(self.diffusion.T - 1, -1, -1):
            particles, log_weights = self.resample(particles, log_weights)
            particles_flat = particles.view(B * K, *sample_shape)
            x_prev = self.proposal(model, particles_flat, t)
            # TODO: proposal and weight need to incorporate the potential in the future, currently potential is 0 so useless in this simple version.
            # potential = self.twist(model, particles_flat, t)
            log_increment_flat = self.weight(model, particles_flat, x_prev, t)
            particles = x_prev.view(B, K, *sample_shape)
            log_weights = log_weights + log_increment_flat.view(B, K)
        # TODO: This is temporary, it returns the first particle since the weights are uniform, in future this should sample one, or return the one with the highest weight etc.
        return particles[:, 0]
