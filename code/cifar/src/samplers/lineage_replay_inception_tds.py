import torch
from tqdm.auto import tqdm

from src.samplers.inception_tds import InceptionTDSSampler


class LineageReplayInceptionTDSSampler(InceptionTDSSampler):
    """Pair guided TDS with an unguided replay of its selected lineage."""

    _NUM_RNG_STREAMS = 4
    _INITIAL_STREAM = 0
    _TRANSITION_STREAM = 1
    _RESAMPLING_STREAM = 2
    _SELECTION_STREAM = 3

    def __init__(self, *, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(seed)

    def _generator(self, device, stream: int) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed * self._NUM_RNG_STREAMS + stream)
        return generator

    def systematic_resample_indices(self, log_weights, generator):
        batch_size, num_particles = log_weights.shape
        weights = torch.softmax(log_weights, dim=1)
        cdf = weights.cumsum(dim=1)
        cdf[:, -1] = 1.0

        offset = torch.rand(
            batch_size,
            1,
            device=log_weights.device,
            dtype=weights.dtype,
            generator=generator,
        ) / num_particles
        points = offset + torch.arange(
            num_particles,
            device=log_weights.device,
            dtype=weights.dtype,
        )[None, :] / num_particles
        return torch.searchsorted(cdf, points, right=False)

    def multinomial_resample_indices(self, log_weights, generator):
        weights = torch.softmax(log_weights, dim=1)
        return torch.multinomial(
            weights,
            num_samples=self.num_particles,
            replacement=True,
            generator=generator,
        )

    def resample_indices(self, log_weights, generator):
        if self.resample_type == "systematic":
            return self.systematic_resample_indices(log_weights, generator)
        if self.resample_type == "multinomial":
            return self.multinomial_resample_indices(log_weights, generator)
        raise ValueError(f"Unsupported resample algorithm {self.resample_type}")

    @staticmethod
    def apply_ancestors(particles, ancestor_indices):
        batch_indices = torch.arange(
            particles.shape[0], device=particles.device
        )[:, None]
        return particles[batch_indices, ancestor_indices]

    @staticmethod
    def trace_lineage(ancestor_history, selected_indices):
        """Trace selected slots back and return transition-noise slots."""
        selected = selected_indices.detach().cpu().long()
        if not ancestor_history:
            return selected, torch.empty((0, selected.shape[0]), dtype=torch.long)

        batch_indices = torch.arange(selected.shape[0])
        noise_slots = [None] * len(ancestor_history)
        slot = selected
        for step in range(len(ancestor_history) - 1, -1, -1):
            noise_slots[step] = slot.clone()
            slot = ancestor_history[step][batch_indices, slot]
        return slot, torch.stack(noise_slots)

    def guided_proposal(
        self,
        model,
        previous,
        t_previous,
        step_size,
        *,
        guidance_active,
        transition_generator,
    ):
        score = self.score_approx(
            model,
            previous,
            t_previous,
            guidance_active=guidance_active,
        )
        mean, variance = self.sde.reverse_transition_params(
            previous,
            t_previous,
            score,
            step_size,
        )
        noise = torch.randn(
            mean.shape,
            device=mean.device,
            dtype=mean.dtype,
            generator=transition_generator,
        )
        std = torch.sqrt(variance).view(-1, 1, 1, 1)
        return (mean + std * noise).detach()

    def replay_unguided(
        self,
        model,
        *,
        boundary_particles,
        boundary_indices,
        noise_slots,
        transition_generator_state,
        timesteps,
        replay_start_step,
        device,
    ):
        batch_size = boundary_particles.shape[0]
        batch_indices = torch.arange(batch_size, device=device)
        boundary_indices = boundary_indices.to(device)
        replay = boundary_particles[batch_indices, boundary_indices].detach()

        if replay_start_step == self.num_steps:
            return replay

        transition_generator = self._generator(device, self._TRANSITION_STREAM)
        transition_generator.set_state(transition_generator_state)

        with torch.no_grad():
            for local_step, step_index in enumerate(
                range(replay_start_step, self.num_steps)
            ):
                t_previous = timesteps[step_index]
                t_current = timesteps[step_index + 1]
                step_size = t_previous - t_current
                t_batch = torch.full(
                    (batch_size,), t_previous.item(), device=device
                )

                score = model(replay, t_batch)
                mean, variance = self.sde.reverse_transition_params(
                    replay,
                    t_batch,
                    score,
                    step_size,
                )

                full_noise = torch.randn(
                    (batch_size, self.num_particles, *self.sample_shape),
                    device=device,
                    dtype=mean.dtype,
                    generator=transition_generator,
                )
                selected_slots = noise_slots[local_step].to(device)
                selected_noise = full_noise[batch_indices, selected_slots]
                std = torch.sqrt(variance).view(batch_size, 1, 1, 1)
                replay = (mean + std * selected_noise).detach()

        return replay

    def sample(self, model, device, progress=True):
        num_particles = self.num_particles
        batch_size = self.num_samples
        initial_generator = self._generator(device, self._INITIAL_STREAM)
        transition_generator = self._generator(device, self._TRANSITION_STREAM)
        resampling_generator = self._generator(device, self._RESAMPLING_STREAM)
        selection_generator = self._generator(device, self._SELECTION_STREAM)

        particles = torch.randn(
            (batch_size, num_particles, *self.sample_shape),
            device=device,
            generator=initial_generator,
        )
        timesteps = torch.linspace(
            self.sde.config.t_max,
            self.sde.config.t_min,
            self.num_steps + 1,
            device=device,
        )
        guidance_activity = self.guidance_activity(timesteps).detach().cpu().tolist()
        replay_start_step = next(
            (index for index, active in enumerate(guidance_activity) if active),
            self.num_steps,
        )
        replay_start_step = min(replay_start_step, self.num_steps)

        initial_t = torch.full(
            (batch_size * num_particles,),
            self.sde.config.t_max,
            device=device,
        )
        log_weights = self.init_weights(
            particles,
            initial_t,
            model,
            guidance_active=guidance_activity[0],
        )
        log_norm = torch.logsumexp(log_weights, dim=1, keepdim=True)
        if not torch.isfinite(log_norm).all():
            raise FloatingPointError("Non-finite initial lineage TDS log norm")
        log_weights = log_weights - log_norm

        iterator = range(self.num_steps)
        if progress:
            iterator = tqdm(
                iterator,
                desc="Lineage-replay Inception TDS sampling",
                leave=True,
            )

        boundary_particles = None
        boundary_transition_state = None
        ancestor_history = []
        resampling_steps = 0

        for step_index in iterator:
            if step_index == replay_start_step:
                boundary_particles = particles.detach().clone()
                boundary_transition_state = transition_generator.get_state()

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
                    ancestor_indices = self.resample_indices(
                        log_weights,
                        resampling_generator,
                    )
                    particles = self.apply_ancestors(particles, ancestor_indices)
                    log_weights = torch.zeros_like(log_weights)
                    resampling_steps += 1
                else:
                    ancestor_indices = torch.arange(
                        num_particles, device=device
                    )[None, :].expand(batch_size, -1)
                if step_index >= replay_start_step:
                    ancestor_history.append(ancestor_indices.detach().cpu().long())
                old_log_weights = log_weights.detach()

            previous = particles.reshape(
                batch_size * num_particles, *self.sample_shape
            )
            current = self.guided_proposal(
                model,
                previous,
                t_previous_flat,
                step_size,
                guidance_active=previous_guidance_active,
                transition_generator=transition_generator,
            )
            incremental_weights = self.log_weight(
                model,
                current,
                previous,
                t_current_flat,
                t_previous_flat,
                step_size,
                current_guidance_active=current_guidance_active,
                prev_guidance_active=previous_guidance_active,
            ).reshape(batch_size, num_particles)

            particles = current.reshape(
                batch_size, num_particles, *self.sample_shape
            ).detach()
            log_weights = (old_log_weights + incremental_weights).detach()
            log_norm = torch.logsumexp(log_weights, dim=1, keepdim=True)
            if not torch.isfinite(log_norm).all():
                raise FloatingPointError("Non-finite lineage TDS log norm")
            log_weights = log_weights - log_norm

        if boundary_particles is None:
            boundary_particles = particles.detach().clone()
            boundary_transition_state = transition_generator.get_state()

        weights = torch.softmax(log_weights, dim=1)
        selected_indices = torch.multinomial(
            weights,
            1,
            generator=selection_generator,
        ).squeeze(1)
        batch_indices = torch.arange(batch_size, device=device)
        guided_samples = particles[batch_indices, selected_indices]

        boundary_indices, noise_slots = self.trace_lineage(
            ancestor_history,
            selected_indices,
        )
        unguided_samples = self.replay_unguided(
            model,
            boundary_particles=boundary_particles,
            boundary_indices=boundary_indices,
            noise_slots=noise_slots,
            transition_generator_state=boundary_transition_state,
            timesteps=timesteps,
            replay_start_step=replay_start_step,
            device=device,
        )

        self.check_tensor("lineage_guided_samples", guided_samples)
        self.check_tensor("lineage_unguided_samples", unguided_samples)
        return {
            "guided_samples": guided_samples.detach(),
            "unguided_samples": unguided_samples.detach(),
            "selected_particle_indices": selected_indices.detach(),
            "selected_boundary_indices": boundary_indices,
            "replay_start_step": replay_start_step,
            "replay_num_steps": self.num_steps - replay_start_step,
            "resampling_steps": resampling_steps,
        }
