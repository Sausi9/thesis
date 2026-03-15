# Cluster Jobs

This folder contains versioned job wrappers for training and evaluation on DTU HPC.

The wrappers:

- run from the repository root
- use `uv` with the checked-in `pyproject.toml` / `uv.lock`
- route caches and temp files into DTU scratch storage when available to avoid filling `$HOME`

## Prerequisites

- The repository is cloned on the cluster and up to date with `git pull --ff-only`
- `uv` is available on the cluster
- You submit the wrappers with `bsub`

Submit from the repository root with `bsub < ...`.

## Train

```bash
bsub < jobs/train_cifar.bsub
```

```bash
bsub < jobs/train_mnist.bsub
```

## Eval

```bash
bsub < jobs/eval_cifar.bsub
```

```bash
bsub < jobs/eval_mnist.bsub
```
