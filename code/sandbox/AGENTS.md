# Repo Guide

## Purpose

This repo started as a small unconditional DDPM implementation for MNIST and CIFAR-10.
It is currently being extended toward:

- TDS (Twisted Diffusion Sampler) style particle-based conditional sampling
- Jeffrey-conditioning based distribution guidance

The current work is exploratory and educational. Preserve clarity over cleverness.

## Current Architecture

### Training and baseline sampling

- `src/engine/train.py`: unconditional DDPM training entry point
- `src/engine/sample.py`: unconditional ancestral DDPM sampling entry point
- `src/engine/eval.py`: unconditional evaluation
- `src/models/unet.py`: epsilon-predicting DDPM U-Net
- `src/data/datasets.py`: MNIST and CIFAR-10 data loading

### Diffusion core

- `src/diffusion/process.py`: owns DDPM math

Important methods:

- `q_sample(...)`: forward noising helper
- `predict_x0_and_eps(...)`: reconstructs `x0_pred` and returns `eps_pred`
- `reverse_mean_variance(...)`: one reverse-step Gaussian parameters
- `sample_reverse_step(...)`: one reverse sampling step

Do not duplicate DDPM formulas in the TDS sampler. Keep diffusion math centralized here.

### Conditioning

- `src/conditioning/base.py`: `ConditioningPotential` interface
- `src/conditioning/null.py`: `NullConditioner`
- `src/conditioning/jeffrey.py`: intended home for Jeffrey-based conditioning logic

The intended design is:

- conditioners score candidate samples
- TDS consumes the conditioner generically
- `NullConditioner` is the no-conditioning baseline

`ConditioningPotential` should stay generic. Avoid coupling it to one specific task.

### TDS work in progress

- `src/diffusion/tds.py`: early `TDSSampler` scaffold
- `src/engine/sample_tds.py`: intended orchestration entry point for TDS experiments

The current `TDSSampler` is not yet a full paper-faithful TDS implementation.

Already sketched:

- particle initialization
- multinomial resampling
- unconditional reverse-kernel proposal
- twist hook via the conditioner
- placeholder zero log-weight update for `NullConditioner`

Still missing / incomplete:

- actual incremental weight formula
- proper twisted proposal using the approximate conditional score
- ESS computation and resample policy
- full `sample(...)` loop
- Jeffrey conditioner implementation
- classifier-backed conditioning for MNIST

## Key Design Decisions

### 1. Keep unconditional training unchanged first

The current plan is to leave DDPM training unconditional and introduce conditioning during inference.

### 2. Use `x_t -> x_{t-1}` indexing

Keep the repo’s existing indexing convention unless there is a strong reason to change it.

### 3. Keep TDS generic

`TDSSampler` should not hard-code class conditioning or Jeffrey logic.
It should depend on the `ConditioningPotential` interface.

### 4. Use log-weights

Particle weights should be maintained in log-space for numerical stability.

### 5. Start simple, then refine

The intended implementation path is:

1. refactor the diffusion API
2. build TDS scaffolding with `NullConditioner`
3. verify particle bookkeeping and resampling
4. add Jeffrey conditioning
5. move toward the full TDS proposal/weight formulas

Do not jump directly to a complicated twisted proposal before the scaffolding is stable.

## Jeffrey Conditioning Notes

The target idea is not standard class conditioning only.
The intended guidance target is Jeffrey-style marginal replacement:

- original label marginal: `p(y)`
- target label marginal: `p*(y)`
- per-sample reweighting factor:
  `psi(x) = sum_y [p*(y) / p(y)] p(y | x)`

Important implication:

- knowing only `p(y)` and `p*(y)` is not enough
- some attribute model such as a classifier `p(y | x)` is still needed

Planned first use case:

- MNIST guidance such as:
  - 100% 7s
  - 50% 6s and 50% 7s

## Practical Advice for Future Agents

- Read `steps.md` first for the implementation roadmap.
- Be careful with tensor shapes. The intended TDS particle layout is `[B, K, C, H, W]`.
- Prefer flattening to `[B*K, C, H, W]` only temporarily when calling networks.
- Keep `Diffusion` responsible for reverse-step math.
- Keep `TDSSampler` responsible for particles, weights, ESS, and resampling.
- Use `NullConditioner` as the baseline when debugging sampler mechanics.
- Do not overwrite or revert user WIP unless explicitly asked.

## Open Work Items

- complete `src/diffusion/tds.py`
- implement `src/conditioning/jeffrey.py`
- add a simple MNIST classifier for `p(y | x)`
- add `sample_tds.py` orchestration
- add evaluation of generated class histograms against target marginals

