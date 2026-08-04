# CIFAR-10 Jeffrey Guidance

CIFAR-10 experiments using a VP-SDE score model, naive score guidance, and TDS.
Two Jeffrey targets are implemented: a scalar brightness marginal and a
2048-dimensional Inception-embedding marginal estimated with a ratio
classifier.

## Setup

Run all commands from this directory:

```bash
cd code/cifar
uv sync --frozen
```

Configuration uses Hydra. Override settings with `group.key=value`; defaults
are under `configs/`. GPU experiments should normally use the LSF jobs in
`jobs/`. Review the variables and `#BSUB` settings near the top of a job file
before submitting it.

## Train and Sample the Base Model

Train locally:

```bash
uv run python -m src.engine.train device=cuda
```

Or submit training on DTU LSF:

```bash
bsub < jobs/train_gpu.bsub
```

Training writes resumable checkpoints to `checkpoints/` and model artifacts to
`artifacts/models/`. Generate unconditional samples from a selected checkpoint
or artifact:

```bash
uv run python -m src.engine.sample \
  device=cuda \
  sampling.artifact_path=checkpoints/RUN/best.pt \
  sampling.weight_type=ema \
  sampling.num_samples=50000 \
  sampling.batch_size=64 \
  sampling.output_name=model_samples_50k
```

The cluster equivalent is `bsub < jobs/sample_gpu.bsub` after setting its
`ARTIFACT_PATH`, sample count, batch size, and output name.

Evaluate a sample payload without FID:

```bash
uv run python -m src.engine.eval \
  sampling.sample_path=runs/samples/SAMPLES.pt \
  eval.fid.enabled=false
```

For FID, use `jobs/eval_gpu.bsub` or run:

```bash
uv run python -m src.engine.eval \
  sampling.sample_path=runs/samples/SAMPLES.pt \
  eval.fid.enabled=true \
  eval.fid.reference=cifar10-train \
  eval.fid.num_samples=50000 \
  eval.fid.min_samples=50000
```

Use `cifar10-test` and `10000` samples for test-set FID.

## Brightness Guidance

Brightness guidance requires unconditional model samples to estimate the
model-induced brightness marginal. Use the same base model for the artifact and
source sample payload.

Naive guidance:

```bash
uv run python -m src.engine.guided_sample \
  device=cuda \
  sampling.artifact_path=checkpoints/RUN/best.pt \
  jeffrey.source_sample_path=runs/samples/model_samples_50k.pt \
  jeffrey.target.mean=0.8 \
  jeffrey.target.variance=0.01 \
  naive_guidance.guidance_coeff=0.5 \
  naive_guidance.guidance_start=0.2
```

TDS:

```bash
uv run python -m src.engine.tds_sample \
  device=cuda \
  sampling.artifact_path=checkpoints/RUN/best.pt \
  jeffrey.source_sample_path=runs/samples/model_samples_50k.pt \
  jeffrey.target.mean=0.8 \
  jeffrey.target.variance=0.01 \
  sampler.num_particles=32 \
  sampler.guidance_scale=1.0 \
  sampler.guidance_start=0.4
```

Cluster jobs are `jobs/guided_gpu.bsub` and `jobs/tds_gpu.bsub`. Sweep jobs are
`jobs/brightness_naive_guidance_sweep_gpu.bsub` and
`jobs/brightness_tds_sweep_gpu.bsub`.

## Inception-Ratio Guidance

### 1. Extract embeddings

Extract FID-compatible Inception features from CIFAR-10 training images and
unconditional model samples:

```bash
uv run python -m src.engine.extract_inception_embeddings \
  device=cuda \
  ratio.generated_sample_path=runs/samples/model_samples_50k.pt
```

On LSF, set `GENERATED_SAMPLE_PATH` in
`jobs/extract_embeddings_gpu.bsub`, then submit it.

### 2. Train the ratio classifier

```bash
uv run python -m src.engine.train_ratio_classifier \
  device=cuda \
  ratio.real_embedding_path=artifacts/embeddings/REAL.pt \
  ratio.generated_embedding_path=artifacts/embeddings/GENERATED.pt \
  ratio.classifier.output_name=inception_ratio
```

The classifier is saved under `artifacts/ratio_classifiers/`. The corresponding
cluster job is `jobs/train_ratio_classifier_gpu.bsub`.

### 3. Run Inception TDS

```bash
uv run python -m src.engine.inception_tds_sample \
  device=cuda \
  jeffrey=inception_ratio \
  sampling.artifact_path=checkpoints/RUN/best.pt \
  jeffrey.ratio_classifier_path=artifacts/ratio_classifiers/inception_ratio/classifier.pt \
  sampling.num_samples=10000 \
  sampling.batch_size=64 \
  sampler.num_particles=4 \
  sampler.guidance_start=0.98 \
  jeffrey.guidance_scale=1.5 \
  sampling.output_name=inception_tds_samples \
  sampling.resume=true
```

For long runs, use resumable shards with
`jobs/inception_tds_fid_shard_gpu.bsub`. Resubmit an incomplete shard with the
same output name, settings, and seed to continue it.

Merge completed shards:

```bash
uv run python -m src.engine.merge_samples \
  'merge.inputs=[runs/samples/SHARD00.pt,runs/samples/SHARD01.pt,runs/samples/SHARD02.pt]' \
  merge.output_name=MERGED_SAMPLES \
  merge.max_samples=10000
```

Run FID on the merged payload with `jobs/eval_gpu.bsub`. After all configured
sweep settings have evaluation metrics, create the sweep plot with:

```bash
uv run python -m src.engine.inception_fid_sweep_plot \
  sweep=inception_tds_fid
```

## Final Diagnostics

Compare unguided and guided sample distributions with CMMD:

```bash
bsub < jobs/eval_cmmd_gpu.bsub
```

Generate coupled guided/unguided lineage pairs, then rank them by Inception
distance:

```bash
bsub < jobs/inception_tds_lineage_paired_gpu.bsub
bsub < jobs/eval_paired_inception_distance_gpu.bsub
```

Set the paired sample path in the second job only after the first job has
completed. The paired outputs include pixel-distance and Inception-distance
figures.

An official DDIM checkpoint can also be sampled as an unconditional baseline
with `jobs/ddim_sample_gpu.bsub`; guided DDIM sampling is not implemented.

## Outputs

- `checkpoints/`: training checkpoints
- `artifacts/models/`: exported model artifacts
- `artifacts/embeddings/`: real and generated Inception features
- `artifacts/ratio_classifiers/`: trained density-ratio classifiers
- `runs/samples/`: generated `.pt` payloads
- `runs/previews/`: sample grids
- `runs/evals/`: FID, brightness, and CMMD results
- `runs/paired/`: paired qualitative comparisons
- `runs/sweeps/`: sweep metrics and plots
