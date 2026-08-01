import torch
from tqdm.auto import tqdm

from src.samplers.inception_tds import InceptionTDSSampler


class PairedInceptionTDSSampler(InceptionTDSSampler):
    """Couple guided TDS particles to counterfactual unguided particles."""

    def systematic_resample_indices(self, log_weights):
        batch_size, num_particles = log_weights.shape
        weights = torch.softmax(log_weights, dim=1)
        cdf = weights.cumsum(dim=1)
        cdf[:, -1] = 1.0

        offset = torch.rand(
            batch_size, 1, device=log_weights.device, dtype=weights.dtype
        ) / num_particles
        points = offset + torch.arange(
            num_particles, device=log_weights.device, dtype=weights.dtype
        )[None, :] / num_particles
        return torch.searchsorted(cdf, points, right=False)

    def multinomial_resample_indices(self, log_weights):
        weights = torch.softmax(log_weights, dim=1)
        return torch.multinomial(
            weights,
            num_samples=self.num_particles,
            replacement=True,
        )

    def resample_indices(self, log_weights):
        if self.resample_type == "systematic":
            return self.systematic_resample_indices(log_weights)
        if self.resample_type == "multinomial":
            return self.multinomial_resample_indices(log_weights)
        raise ValueError(f"Unsupported resample algorithm {self.resample_type}")

    @staticmethod
    def apply_ancestors(particles, ancestor_indices):
        batch_indices = torch.arange(
            particles.shape[0], device=particles.device
        )[:, None]
        return particles[batch_indices, ancestor_indices]

    def paired_proposal(
        self,
        model,
        guided_previous,
        unguided_previous,
        t_previous,
        step_size,
        *,
        guidance_active,
    ):
        guided_score = self.score_approx(
            model,
            guided_previous,
            t_previous,
            guidance_active=guidance_active,
        )
        guided_mean, proposal_variance = self.sde.reverse_transition_params(
            guided_previous,
            t_previous,
            guided_score,
            step_size,
        )

        if guidance_active:
            with torch.no_grad():
                unguided_score = model(unguided_previous, t_previous)
                unguided_mean, _ = (
                    self.sde.reverse_transition_params(
                        unguided_previous,
                        t_previous,
                        unguided_score,
                        step_size,
                    )
                )
        else:
            # The banks are identical before guidance. Reusing the same mean avoids
            # an unnecessary model evaluation and keeps the coupling exact.
            unguided_mean = guided_mean

        self.check_tensor("paired_guided_mean", guided_mean)
        self.check_tensor("paired_unguided_mean", unguided_mean)
        self.check_tensor("paired_proposal_variance", proposal_variance)

        shared_noise = torch.randn_like(guided_mean)
        proposal_std = torch.sqrt(proposal_variance).view(-1, 1, 1, 1)
        guided_next = guided_mean + proposal_std * shared_noise
        unguided_next = unguided_mean + proposal_std * shared_noise
        return guided_next.detach(), unguided_next.detach()

    def sample(self, model, device, progress=True):
        num_particles = self.num_particles
        batch_size = self.num_samples
        guided_particles = self.init_particles(batch_size, device)
        unguided_particles = guided_particles.clone()

        timesteps = torch.linspace(
            self.sde.config.t_max,
            self.sde.config.t_min,
            self.num_steps + 1,
            device=device,
        )
        guidance_activity = self.guidance_activity(timesteps).detach().cpu().tolist()

        initial_t = torch.full(
            (batch_size * num_particles,),
            self.sde.config.t_max,
            device=device,
        )
        log_weights = self.init_weights(
            guided_particles,
            initial_t,
            model,
            guidance_active=guidance_activity[0],
        )
        log_norm = torch.logsumexp(log_weights, dim=1, keepdim=True)
        if not torch.isfinite(log_norm).all():
            raise FloatingPointError("Non-finite initial paired TDS log norm")
        log_weights = log_weights - log_norm

        iterator = range(self.num_steps)
        if progress:
            iterator = tqdm(iterator, desc="Paired Inception TDS sampling", leave=True)

        resampling_steps = 0
        for step_index in iterator:
            t_previous = timesteps[step_index]
            t_current = timesteps[step_index + 1]
            step_size = t_previous - t_current
            previous_guidance_active = guidance_activity[step_index]
            current_guidance_active = guidance_activity[step_index + 1]

            t_previous_flat = torch.full(
                (batch_size * num_particles,), t_previous.item(), device=device
            )
            t_current_flat = torch.full(
                (batch_size * num_particles,), t_current.item(), device=device
            )

            with torch.no_grad():
                mean_ess = self.ess(log_weights).mean()
                should_resample = (
                    not self.adaptive_resampling
                    or mean_ess < self.ess_threshold * num_particles
                )
                if should_resample:
                    ancestor_indices = self.resample_indices(log_weights)
                    guided_particles = self.apply_ancestors(
                        guided_particles, ancestor_indices
                    )
                    unguided_particles = self.apply_ancestors(
                        unguided_particles, ancestor_indices
                    )
                    log_weights = torch.zeros_like(log_weights)
                    resampling_steps += 1
                old_log_weights = log_weights.detach()

            guided_previous = guided_particles.reshape(
                batch_size * num_particles, *self.sample_shape
            )
            unguided_previous = unguided_particles.reshape(
                batch_size * num_particles, *self.sample_shape
            )
            guided_current, unguided_current = self.paired_proposal(
                model,
                guided_previous,
                unguided_previous,
                t_previous_flat,
                step_size,
                guidance_active=previous_guidance_active,
            )
            incremental_weights = self.log_weight(
                model,
                guided_current,
                guided_previous,
                t_current_flat,
                t_previous_flat,
                step_size,
                current_guidance_active=current_guidance_active,
                prev_guidance_active=previous_guidance_active,
            ).reshape(batch_size, num_particles)

            guided_particles = guided_current.reshape(
                batch_size, num_particles, *self.sample_shape
            ).detach()
            unguided_particles = unguided_current.reshape(
                batch_size, num_particles, *self.sample_shape
            ).detach()
            log_weights = (old_log_weights + incremental_weights).detach()
            log_norm = torch.logsumexp(log_weights, dim=1, keepdim=True)
            if not torch.isfinite(log_norm).all():
                raise FloatingPointError("Non-finite paired TDS log norm")
            log_weights = log_weights - log_norm

        weights = torch.softmax(log_weights, dim=1)
        selected_indices = torch.multinomial(weights, 1).squeeze(1)
        batch_indices = torch.arange(batch_size, device=device)
        return {
            "guided_samples": guided_particles[batch_indices, selected_indices],
            "unguided_samples": unguided_particles[batch_indices, selected_indices],
            "selected_particle_indices": selected_indices,
            "resampling_steps": resampling_steps,
        }
