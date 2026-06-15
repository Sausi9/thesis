# CIFAR-10 TDS

CIFAR-10 score-SDE experiments for Jeffrey-guided TDS generation.

This project mirrors the structure of `code/bivariate`, but uses image-shaped
particles and a scalar brightness marginal:

```text
f(x) = mean((x + 1) / 2)
```

The v1 implementation supports:

- unconditional VP-SDE score-model training
- unconditional image generation
- Jeffrey-guided TDS generation toward a Gaussian brightness marginal
- sample evaluation with image previews and brightness histograms

## Commands

```bash
uv run python -m src.engine.train
uv run python -m src.engine.sample
uv run python -m src.engine.tds_sample
uv run python -m src.engine.eval
```

For quick smoke checks, override the heavy defaults, for example:

```bash
uv run python -m src.engine.sample sampling.num_samples=4 sampling.batch_size=2 sampling.num_steps=2
```
