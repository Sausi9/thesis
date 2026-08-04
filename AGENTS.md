# Thesis Agent Guide

## Scope and purpose

This repository is an MSc thesis project about distribution-guided generation with
diffusion models. The central question is whether Jeffrey's rule can turn an
unconditional diffusion model into a sampler whose generated attributes follow a
desired marginal distribution, with Twisted Diffusion Sampler (TDS) as the main
sampling method.

Read this file before changing code or thesis text. Treat the current source and
saved payload metadata as authoritative for implementation details; the short
subproject READMEs and some prose in `writing/` can lag behind the code. When a
claim matters to the thesis, verify it against both the implementation and the
latest relevant text/results rather than inferring it from a default YAML value.

## Research model

Jeffrey's rule replaces an undesired marginal while preserving the corresponding
conditional structure:

```text
p*(z_I, z_J) = p(z_I | z_J) p*(z_J)
              = p(z_I, z_J) p*(z_J) / p(z_J).
```

The density ratio `p*(z_J) / p(z_J)` is therefore the clean-data potential used by
importance resampling, naive score guidance, and TDS. TDS is an SMC correction of
guided proposals: particles are proposed using the gradient of an approximate
time-dependent log potential, then incremental log weights correct for proposing
from the approximate twisted transition rather than the original reverse kernel.
The correction is asymptotic in the particle count; it does not remove
finite-particle degeneracy or rescue a proposal that misses the target.

The important distinction from ordinary conditional TDS is that this project is
not generally conditioning on one fixed observation with `p(y | x0)`. It is
changing a marginal distribution. In the deterministic-attribute experiments the
clean potential is directly a density ratio. In the Inception experiment, a
balanced classifier estimates the target/source density ratio in feature space.

## Repository map

- `writing/`: LaTeX thesis. `Chapters/02_Background.tex` contains the mathematical
  Jeffrey/TDS construction; `03_Methodology.tex` and `04_Results.tex` are evolving
  and may contain TODOs or lag behind decisions recorded in experiments.
- `code/bivariate/`: analytically tractable two-dimensional Gaussian validation.
- `code/cifar/`: CIFAR-10 image experiments, brightness guidance, the Inception
  density-ratio pipeline, FID evaluation, and an unconditional DDIM baseline.
- `hpc_runs/`, `runs/`, `checkpoints/`, `artifacts/`, and `data/`: generated or
  machine-local state. Preserve it, do not casually delete it, and do not commit
  large outputs unless explicitly requested.

`code/bivariate` and `code/cifar` are separate UV projects with separate
`pyproject.toml`, `uv.lock`, `.venv`, Hydra configs, and a top-level Python package
named `src`. Run commands from the relevant subproject directory. There is no root
Python environment that safely represents both projects.

## Shared diffusion conventions

Both main implementations train a neural network to predict the continuous-time
score directly under a variance-preserving SDE. Training uses denoising score
matching with the equivalent objective
`mean((score_pred * std + epsilon)^2)`. Sampling discretizes the reverse VP-SDE
with Euler-Maruyama. The transition variance for one reverse step is
`beta(t) * step_size`.

The denoised estimate used in the guidance potential is

```text
x0_hat = (x_t + sigma(t)^2 * score(x_t, t)) / alpha(t).
```

Early in the reverse process this estimate is noisy. Delaying or ramping guidance
has been one of the most consequential experimental choices. `guidance_start` is
the fraction of the reverse trajectory already completed, measured from noisy
`t_max` toward clean `t_min`:

- `guidance_ramp: null`: full guidance throughout.
- `linear`: ramp from zero at the start of the reverse process to full strength.
- `delayed_linear`: zero before `guidance_start`, then linearly ramp to full
  strength over the remaining steps.
- `delayed_discrete`: zero before `guidance_start`, then jump to full strength.

For TDS, the ramp and guidance scale multiply the log potential itself, so they
affect both proposal guidance and the SMC weight sequence. Do not change only one
side. Preserve log-space calculations, the shared proposal/original variance, and
the transition ratio identity used in `log_weight` unless deliberately revising
the mathematics. The weight must evaluate the exact transition that was proposed;
never draw a second sample inside the weight calculation.

The TDS implementations support multinomial and systematic resampling, either at
every step or adaptively when the mean batch ESS falls below
`ess_threshold * K`. This is currently one resampling decision based on mean ESS
across the batch, not a separate decision per target sample. Changing that is an
algorithmic change, not cleanup. Guidance gradients are finite-checked and clipped
by `max_guidance_grad_norm`.

## Bivariate implementation

### Experiment and exact baseline

The configured source distribution is

```text
(x, y) ~ N([0, 0], [[1.0, 0.8], [0.8, 1.5]]).
```

The default updated coordinate is `y` (`updated_dim: 1`). The principal reported
targets use Gaussian `y` marginals with means 2 and 6 and variance 0.25. Because
the source joint and target marginal are Gaussian, the Jeffrey-updated joint is
available in closed form. `src/jeffrey/update.py` both computes its mean/covariance
and samples it by drawing the updated coordinate from `p*(y)` and the retained
coordinate from the original `p(x | y)`.

The unconditional source path plus the four evaluated methods are:

- `src.engine.sample`: unconditional model samples.
- `src.engine.exact_jeffrey_sample`: exact Jeffrey reference samples.
- `src.engine.importance_resampling`: terminal ratio weighting and resampling.
- `src.engine.guided_sample`: naive reverse-SDE score guidance without SMC weights.
- `src.engine.tds_sample`: Jeffrey-guided TDS; this is the main method despite
  appearing as a fifth entrypoint because unconditional sampling supplies inputs.

Importance resampling deliberately uses the analytic original dataset marginal in
its denominator. Bivariate TDS instead estimates the model-induced marginal
`p_theta(y)` from an unconditional payload whose `sample_type` is exactly `model`.
This distinction is intentional and scientifically important. Always use an
unconditional source sample produced by the same model artifact as the guided run.

`src/samplers/tds.py` has two twist modes:

- `tractable`: evaluates `log p*(y_hat) - log p_theta(y_hat)` at the denoised
  coordinate. This is the practical Jeffrey twist.
- `optimal`: propagates Gaussian source/target parameters through the forward SDE
  and uses their time-marginal density ratio. In the current engine the source
  Gaussian is fitted to unconditional model samples while the updated Gaussian is
  calculated from the configured dataset joint. It is a toy-only oracle-style
  diagnostic, not a generally available twist.

The reported final bivariate TDS runs used roughly `K=2048`, `T=1000`, and 12,000
samples; older exploratory runs used much larger `K` (including 16,384). Do not
assume the current Hydra defaults reproduce a reported figure. Read the saved
payload's `config`, `run_label`, and explicit metadata.

The durable result narrative is: importance resampling works under the mild shift
but degenerates for the non-overlapping mean-6 target; naive guidance and TDS work
well for both; TDS better preserves the full covariance structure in the severe
shift, at substantially higher computational cost. The exact target makes this a
validation/sanity-check experiment, not merely a visualization.

### Bivariate commands

From `code/bivariate`:

```bash
uv run python -m src.engine.train
uv run python -m src.engine.sample
uv run python -m src.engine.exact_jeffrey_sample
uv run python -m src.engine.importance_resampling
uv run python -m src.engine.guided_sample
uv run python -m src.engine.tds_sample
uv run python -m src.engine.eval sampling.sample_path=runs/samples/FILE.pt
```

`run_sampling_suite.sh` runs unconditional, exact, importance-resampling, and naive
guidance paths plus their evaluations; it does not run TDS. The resampling sweep is
defined in `experiments/tds_resampling_fixed.yaml` and submitted through
`src.engine.submit_experiments`.

## CIFAR-10 implementation

### Base model and image contract

CIFAR images have shape `[N, 3, 32, 32]` and are stored in model space `[-1, 1]`.
Brightness is the scalar

```text
mean((image + 1) / 2)
```

over channels and pixels, so it is normally in `[0, 1]` for clamped images.

The default model is a continuous VP-SDE `ScoreUNet` with spatial attention. The
training path includes learning-rate warmup, gradient clipping, validation,
training previews, and EMA. Sampling defaults to requesting EMA weights; payloads
record both the requested and actually loaded weight type. Preserve this metadata.

`src.engine.sample` has two unconditional backends:

- `sampling.backend=score_sde` (default): the project's VP-SDE model, saved with
  `sample_type: model`.
- `sampling.backend=ddim`: the official CIFAR DDIM architecture/checkpoint baseline,
  saved with `sample_type: ddim_model`.

DDIM is currently an unconditional sampling/FID baseline only. Brightness and
Inception TDS/guidance remain score-SDE code paths. The distinct DDIM sample type
prevents automatic source selection from mixing an incompatible DDIM checkpoint
with the score-SDE samplers. Do not make DDIM a TDS backend by a superficial config
switch; it requires a separate discrete proposal and weight derivation.

### Brightness experiment

`src.engine.tds_sample` estimates the original Gaussian brightness marginal from
an unconditional `sample_type: model` payload and guides toward the configured
Gaussian target. `src.engine.guided_sample` implements the corresponding naive
baseline. Sweeps live in `brightness_tds_sweep.py` and
`brightness_naive_guidance_sweep.py`; their output directories are created with
`exist_ok=False`, so every sweep needs a unique `sweep.output_name`.

The thesis brightness target is mean 0.8 and standard deviation 0.1 (variance
0.01). Current thesis context records delayed-discrete guidance for the naive
baseline and delayed-linear guidance for the selected TDS result. Several nearby
start/scale settings were explored, and chat recommendations changed as larger
runs arrived; use the exact final payload rather than a remembered recommendation
when reporting hyperparameters. More samples reduce Monte Carlo uncertainty; they
do not necessarily remove systematic guidance error.

### Inception density-ratio experiment

This experiment adapts the real-vs-generated Inception-distribution guidance idea
from the referenced "Towards More General..." work, but the thesis contribution is
to apply TDS rather than merely reproduce its naive guidance. The data flow is:

1. `extract_inception_embeddings.py` extracts torch-fidelity
   `inception-v3-compat` layer-2048 features for real CIFAR-10 training images and
   unconditional score-SDE samples.
2. `train_ratio_classifier.py` balances the two classes, labels real as 1 and
   generated as 0, standardizes features from the training split, and trains a
   linear binary classifier.
3. With balanced class priors, the classifier logit estimates
   `log p_real(feature) - log p_model(feature)`, the target/source log density
   ratio in Inception space.
4. `InceptionRatioPotential` recreates the same FID-compatible preprocessing with
   continuous differentiable inputs, applies the saved standardization and linear
   classifier, and optionally clips logits.
5. `inception_tds_sample.py` evaluates that log-ratio on `x0_hat`, differentiates it
   through the Inception network with respect to the noisy image, and uses it as
   the TDS log potential.

`inception_guided_sample.py` exists as an implemented score-guidance baseline, but
the current thesis scope does not include an independently run Inception
naive-guidance experiment; the cited paper supplies that comparison context. Do not
write the implementation's mere existence as a completed thesis experiment.

The Inception TDS job is extremely expensive: every reverse step runs the U-Net,
the differentiable Inception potential/gradient, and SMC bookkeeping for `B*K`
particles. The shard entrypoint supports atomic incremental saving and resume.
With `sampling.resume=true`, use a fixed `sampling.output_name`; a resumed shard
must keep the same artifact, classifier, seed, batch size, particle count, steps,
guidance, resampling, preprocessing, and clipping settings. Do not delete a partial
payload before resubmission. The code advances reproducible batch seeds as
`base_seed + num_batches_completed`.

The current FID sweep config names 12 settings: starts `[0.4, 0.6, 0.8]` crossed
with scales `[0.25, 0.5, 1, 2]`, normally three complete shards totaling 10,000
samples per setting. `merge_samples.py` rejects incomplete shards by default and
checks shared scientific metadata. The plotter expects exact filename-safe names
such as `start0p4_scale1`, so keep job, merge, eval, and plot names aligned. As of
the latest project context, the final Inception TDS schedule/result was still being
selected; do not silently promote the current delayed-discrete YAML default to a
final thesis conclusion.

### CIFAR commands

From `code/cifar`:

```bash
uv run python -m src.engine.train
uv run python -m src.engine.sample
uv run python -m src.engine.sample sampling.backend=ddim
uv run python -m src.engine.guided_sample
uv run python -m src.engine.tds_sample
uv run python -m src.engine.extract_inception_embeddings
uv run python -m src.engine.train_ratio_classifier
uv run python -m src.engine.inception_tds_sample jeffrey=inception_ratio
uv run python -m src.engine.eval sampling.sample_path=runs/samples/FILE.pt eval.fid.enabled=false
uv run python -m src.engine.eval_inception_ratio sampling.sample_path=runs/samples/FILE.pt jeffrey=inception_ratio
uv run python -m src.engine.merge_samples 'merge.inputs=[A.pt,B.pt,C.pt]' merge.output_name=MERGED
uv run python -m src.engine.inception_fid_sweep_plot sweep=inception_tds_fid
```

Normal evaluation always writes a preview, brightness histogram, and metrics JSON.
FID is optional and enforces the configured minimum sample count. The committed
`eval_gpu.bsub` wrapper forces FID on; for brightness-only evaluation explicitly
use `eval.fid.enabled=false` or the direct command above.

## Saved payloads are interfaces

Do not treat `.pt` files as unstructured tensors. Engines, evaluation,
auto-selection, merging, resume, naming, and thesis reproducibility depend on metadata
such as:

```text
sample_type, config, artifact_path, loaded_weight_type,
target_marginal/original_marginal, guidance_* and resampling fields,
complete, num_completed, resume_compatibility
```

Preserve existing keys when extending payloads. New derived generators should get
a specific `sample_type`; do not label them `model` unless they are valid
unconditional score-SDE sources for all auto-selecting Jeffrey pipelines. Automatic
"latest" discovery is based on modification time and is convenient but fragile.
For reproducible or final experiments, pass explicit artifact, source sample, ratio
classifier, sample, and output paths.

## Cluster workflow

Long CIFAR and high-particle bivariate runs are intended for DTU's LSF cluster.
The `.bsub` files define resources and queues; the paired `jobs/*.sh` wrappers set
scratch-backed UV/cache/temp directories, run `uv sync --frozen`, and execute the
Hydra module from the subproject root. Queue availability and access change, so do
not document a queue as universally available and do not submit, kill, duplicate,
or move jobs unless the user asks.

Use distinct output names for concurrent jobs. Never run two active jobs that write
the same non-resumable output. For a resumable Inception shard, resubmit the same
job identity and scientific settings. Before merging, inspect the payload rather
than inferring completion from a progress-log line count: require the requested
shape/count and `complete=True`.

## Change and validation rules

- Keep algorithmic code in `src/samplers` and orchestration/config/path/payload
  logic in `src/engine`.
- Hydra config is the source of defaults. Prefer overrides for one-off smoke tests;
  edit YAML when defining a reproducible named experiment. Remember that `.bsub`
  environment variables and `EXTRA_OVERRIDES` can override YAML.
- When changing shared VP-SDE/TDS logic, inspect both subprojects and both CIFAR
  potentials (brightness and Inception). Port only shape-appropriate changes; do
  not assume 2D and image reductions/broadcasts are interchangeable.
- Preserve tensor shapes and model space: bivariate `[N, 2]`; CIFAR
  `[N, 3, 32, 32]` in `[-1, 1]`.
- Freeze model and feature-extractor parameters during sampling while retaining
  gradients with respect to the noisy image required for guidance.
- Do not rewrite mathematical behavior as style cleanup. Guidance scheduling,
  detach points, transition-ratio algebra, ESS aggregation, final weighted particle
  selection, source-marginal estimation, and sample types are scientific choices.
- There is currently no conventional unit-test suite. At minimum, run
  `uv run python -m compileall src` in each affected subproject. For behavioral
  changes, add a tiny explicit smoke run (few samples, particles, and steps) with
  explicit artifact/source/classifier and a unique ignored output name. Do not run
  default training, 1,000-step TDS, FID, downloads, or cluster submissions as a
  routine validation step.
- If checkpoints, datasets, network downloads, GPU access, or cluster state are
  unavailable, report the unrun validation precisely; never claim a full experiment
  passed based only on compilation.
- Preserve unrelated working-tree changes and generated results. Never clean
  `runs/`, `artifacts/`, `checkpoints/`, `data/`, `hpc_runs/`, or `tmp/` as part of
  normal code work.

For thesis writing, separate mathematical definitions, methodology choices,
observed results, and discussion. Do not present exploratory settings as final,
do not turn finite-sample differences into unsupported superiority claims, and do
not say that more samples remove systematic approximation error. Keep the user's
plain, direct writing voice while correcting mathematical or grammatical errors.
