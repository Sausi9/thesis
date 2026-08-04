# Bivariate Jeffrey Guidance

Bivariate Gaussian experiments comparing exact Jeffrey sampling, importance
resampling, naive score guidance, and the Twisted Diffusion Sampler (TDS).

## Setup

Run all commands from this directory:

```bash
cd code/bivariate
uv sync --frozen
```

Configuration uses Hydra. Override any setting with `group.key=value` on the
command line. The main defaults are under `configs/`.

## Basic Workflow

Train the score model:

```bash
uv run python -m src.engine.train
```

Training writes checkpoints to `checkpoints/` and reusable model artifacts to
`artifacts/models/`. Generate unconditional samples from a selected artifact:

```bash
uv run python -m src.engine.sample \
  sampling.artifact_path=artifacts/models/MODEL_best.pt \
  sampling.output_name=model_samples
```

Evaluate any saved sample payload:

```bash
uv run python -m src.engine.eval \
  sampling.sample_path=runs/samples/model_samples.pt
```

## Jeffrey Sampling Methods

Set the target marginal with `jeffrey.target.mean` and
`jeffrey.target.variance`. The updated coordinate is selected with
`jeffrey.updated_dim`.

Analytic Jeffrey sampling:

```bash
uv run python -m src.engine.exact_jeffrey_sample \
  jeffrey.target.mean=6.0 \
  jeffrey.target.variance=0.25
```

Importance resampling from an unconditional sample payload:

```bash
uv run python -m src.engine.importance_resampling \
  jeffrey.source_sample_path=runs/samples/model_samples.pt
```

Naive score guidance:

```bash
uv run python -m src.engine.guided_sample \
  sampling.artifact_path=artifacts/models/MODEL_best.pt \
  naive_guidance.guidance_scale=0.5 \
  naive_guidance.guidance_start=0.6
```

TDS generation:

```bash
uv run python -m src.engine.tds_sample \
  sampling.artifact_path=artifacts/models/MODEL_best.pt \
  jeffrey.source_sample_path=runs/samples/model_samples.pt \
  sampler.num_particles=256 \
  sampler.guidance_scale=1.0 \
  sampler.guidance_start=0.6
```

The source samples must come from the same model artifact used for guided
sampling. They estimate the model-induced marginal required by the Jeffrey
density ratio.

## Guidance Sweep

The two-stage naive-guidance and TDS sweep is configured in
`configs/sweep/guidance.yaml`.

```bash
uv run python -m src.engine.guidance_sweep \
  sampling.artifact_path=artifacts/models/MODEL_best.pt \
  jeffrey.source_sample_path=runs/samples/model_samples.pt \
  sweep.output_name=guidance_calibration
```

After calibration, run confirmation using its result file:

```bash
uv run python -m src.engine.guidance_sweep \
  sampling.artifact_path=artifacts/models/MODEL_best.pt \
  jeffrey.source_sample_path=runs/samples/model_samples.pt \
  sweep.stage=confirmation \
  sweep.calibration_results_path=runs/sweeps/guidance/guidance_calibration/results.json \
  sweep.output_name=guidance_confirmation
```

For DTU LSF, review the variables in the job files and submit with:

```bash
bsub < jobs/tds_gpu.bsub
bsub < jobs/guidance_sweep_gpu.bsub
bsub < jobs/guidance_sweep_confirmation_gpu.bsub
```

## Outputs

- `artifacts/models/`: final and best model artifacts
- `checkpoints/`: resumable training checkpoints
- `runs/samples/`: generated `.pt` payloads
- `runs/evals/`: evaluation figures
- `runs/sweeps/`: sweep metrics and plots
