# Bivariate

Bivariate Gaussian toy example.

## TDS Resampling Sweep

The fixed-size resampling comparison is defined in `experiments/tds_resampling_fixed.yaml`.
It launches four jobs:

- multinomial, always resample
- multinomial, ESS-adaptive resampling
- systematic, always resample
- systematic, ESS-adaptive resampling

Preview the cluster commands:

```bash
uv run python -m src.engine.submit_experiments --dry-run
```

Submit all four jobs:

```bash
uv run python -m src.engine.submit_experiments
```

Submit only selected variants:

```bash
uv run python -m src.engine.submit_experiments --only systematic_adaptive multinomial_always
```

Submitted runs are recorded in `runs/experiments/tds_resampling_fixed/manifest.csv`
and `runs/experiments/tds_resampling_fixed/manifest.jsonl`.
