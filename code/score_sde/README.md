# Score SDE

Reusable scaffolding copied from `sandbox` for an SDE-based score model implementation.

Included:

- dataset loading
- artifact lookup
- U-Net backbone
- configs and cluster job wrappers
- train/sample/eval entrypoints with explicit SDE placeholders

Not included:

- DDPM discrete diffusion process
- DDPM training loss
- DDPM ancestral sampler
