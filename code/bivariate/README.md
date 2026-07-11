# Bivariate

Bivariate Gaussian toy example.

## Guidance Scale Sweep

Naive guidance and TDS both expose `guidance_scale`. For TDS the scale multiplies
the scheduled log twist, so it affects both proposal guidance and SMC weights.
Scale `1` is the untempered Jeffrey potential; other scales define tempered
potentials. The deprecated naive-guidance key `guidance_coeff` remains available
as an alias.

The unified two-stage experiment is configured in `configs/sweep/guidance.yaml`.
It requires an explicit model artifact and unconditional sample payload produced
by that same artifact.

Run the calibration grid:

```bash
uv run python -m src.engine.guidance_sweep \
  sampling.artifact_path=artifacts/models/MODEL_best.pt \
  jeffrey.source_sample_path=runs/samples/MODEL_model_samples.pt \
  sweep.output_name=guidance_calibration
```

Resume the same calibration output:

```bash
uv run python -m src.engine.guidance_sweep \
  sampling.artifact_path=artifacts/models/MODEL_best.pt \
  jeffrey.source_sample_path=runs/samples/MODEL_model_samples.pt \
  sweep.output_name=guidance_calibration \
  sweep.resume=true
```

After calibration completes, automatically confirm the best scale/start for each
method and target:

```bash
uv run python -m src.engine.guidance_sweep \
  sampling.artifact_path=artifacts/models/MODEL_best.pt \
  jeffrey.source_sample_path=runs/samples/MODEL_model_samples.pt \
  sweep.stage=confirmation \
  sweep.calibration_results_path=runs/sweeps/guidance/guidance_calibration/results.json \
  sweep.output_name=guidance_confirmation
```

The LSF jobs are self-contained and require no exported shell variables. Review
the settings blocks near the top of each file, then submit calibration and
confirmation respectively:

```bash
bsub < jobs/guidance_sweep_gpu.bsub
bsub < jobs/guidance_sweep_confirmation_gpu.bsub
```

The committed defaults run only the mean-6 target with the matching local 20260504
artifact/source pair. To resume either output, change its `RESUME` assignment to
`true` without changing any other scientific setting. No job is submitted by the
sweep Python command itself.

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
