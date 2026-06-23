from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

cache_root = Path(tempfile.gettempdir()) / "cifar_ratio_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ratio.classifier import LinearRatioClassifier, standardize
from src.utils import resolve_device, resolve_path


def find_latest_embedding(embedding_dir: Path, source_kind: str) -> Path:
    matches = sorted(
        embedding_dir.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in matches:
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if payload.get("source_kind") == source_kind:
            return path
    raise FileNotFoundError(f"No source_kind={source_kind!r} embedding artifacts found in {embedding_dir}.")


def load_embedding(path: Path, expected_kind: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if payload.get("source_kind") != expected_kind:
        raise ValueError(
            f"Expected {expected_kind!r} embedding artifact, got {payload.get('source_kind')!r} at {path}."
        )
    features = payload.get("features")
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError(f"Embedding artifact {path} must contain a 2D tensor under key 'features'.")
    return payload


def split_balanced_features(
    *,
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> dict:
    n = min(int(real_features.shape[0]), int(generated_features.shape[0]))
    if n < 2:
        raise ValueError("Need at least two samples from each class to train the ratio classifier.")
    real_features = real_features[:n].float()
    generated_features = generated_features[:n].float()

    generator = torch.Generator().manual_seed(int(seed))
    real_perm = torch.randperm(n, generator=generator)
    generated_perm = torch.randperm(n, generator=generator)
    real_features = real_features[real_perm]
    generated_features = generated_features[generated_perm]

    val_n = int(round(n * float(val_fraction)))
    val_n = max(1, min(n - 1, val_n))
    train_n = n - val_n

    train_real = real_features[:train_n]
    val_real = real_features[train_n:]
    train_generated = generated_features[:train_n]
    val_generated = generated_features[train_n:]

    train_x = torch.cat([train_real, train_generated], dim=0)
    train_y = torch.cat(
        [
            torch.ones(train_real.shape[0]),
            torch.zeros(train_generated.shape[0]),
        ],
        dim=0,
    )
    val_x = torch.cat([val_real, val_generated], dim=0)
    val_y = torch.cat(
        [
            torch.ones(val_real.shape[0]),
            torch.zeros(val_generated.shape[0]),
        ],
        dim=0,
    )

    train_perm = torch.randperm(train_x.shape[0], generator=generator)
    val_perm = torch.randperm(val_x.shape[0], generator=generator)
    return {
        "train_x": train_x[train_perm],
        "train_y": train_y[train_perm],
        "val_x": val_x[val_perm],
        "val_y": val_y[val_perm],
        "num_per_class": n,
        "train_per_class": train_n,
        "val_per_class": val_n,
    }


@torch.no_grad()
def evaluate(model, x: torch.Tensor, y: torch.Tensor, criterion, device: torch.device) -> dict:
    model.eval()
    logits = model(x.to(device)).detach().cpu()
    labels = y.float()
    loss = criterion(logits, labels).item()
    probs = torch.sigmoid(logits)
    predictions = (probs >= 0.5).float()
    accuracy = (predictions == labels).float().mean().item()
    real_logits = logits[labels == 1]
    generated_logits = logits[labels == 0]
    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "real_logit_mean": float(real_logits.mean()) if real_logits.numel() else float("nan"),
        "generated_logit_mean": float(generated_logits.mean())
        if generated_logits.numel()
        else float("nan"),
        "logits": logits,
        "labels": labels,
    }


def save_logit_histogram(metrics: dict, output_path: Path) -> None:
    logits = metrics["logits"]
    labels = metrics["labels"]
    real_logits = logits[labels == 1]
    generated_logits = logits[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(
        generated_logits.numpy(),
        bins=40,
        alpha=0.6,
        density=True,
        label="generated",
        color="#2563eb",
    )
    ax.hist(
        real_logits.numpy(),
        bins=40,
        alpha=0.6,
        density=True,
        label="real train",
        color="#f97316",
    )
    ax.axvline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    ax.set_xlabel("classifier logit")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_output_dir(project_root: Path, cfg: DictConfig) -> Path:
    output_dir = resolve_path(project_root, str(cfg.ratio.classifier_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = cfg.ratio.classifier.output_name
    if output_name is None:
        output_name = f"inception_ratio_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return output_dir / str(output_name)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    classifier_cfg = cfg.ratio.classifier

    embedding_dir = resolve_path(project_root, str(cfg.ratio.embedding_dir))
    real_embedding_path = (
        find_latest_embedding(embedding_dir, "real")
        if cfg.ratio.real_embedding_path is None
        else resolve_path(project_root, str(cfg.ratio.real_embedding_path))
    )
    generated_embedding_path = (
        find_latest_embedding(embedding_dir, "generated")
        if cfg.ratio.generated_embedding_path is None
        else resolve_path(project_root, str(cfg.ratio.generated_embedding_path))
    )

    real_payload = load_embedding(real_embedding_path, "real")
    generated_payload = load_embedding(generated_embedding_path, "generated")
    real_features = real_payload["features"].float()
    generated_features = generated_payload["features"].float()

    if int(real_features.shape[1]) != int(classifier_cfg.embedding_dim):
        raise ValueError(
            f"Expected real embedding_dim={classifier_cfg.embedding_dim}, got {real_features.shape[1]}."
        )
    if int(generated_features.shape[1]) != int(classifier_cfg.embedding_dim):
        raise ValueError(
            f"Expected generated embedding_dim={classifier_cfg.embedding_dim}, got {generated_features.shape[1]}."
        )

    split = split_balanced_features(
        real_features=real_features,
        generated_features=generated_features,
        val_fraction=float(classifier_cfg.val_fraction),
        seed=int(classifier_cfg.seed),
    )

    train_mean = split["train_x"].mean(dim=0, keepdim=True)
    train_std = split["train_x"].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = standardize(split["train_x"], train_mean, train_std)
    val_x = standardize(split["val_x"], train_mean, train_std)
    train_y = split["train_y"]
    val_y = split["val_y"]

    generator = torch.Generator().manual_seed(int(classifier_cfg.seed))
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=int(classifier_cfg.batch_size),
        shuffle=True,
        generator=generator,
    )

    model = LinearRatioClassifier(int(classifier_cfg.embedding_dim)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(classifier_cfg.lr),
        weight_decay=float(classifier_cfg.weight_decay),
    )

    history = []
    for epoch in range(1, int(classifier_cfg.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch_x.shape[0])
            total_count += int(batch_x.shape[0])

        val_metrics = evaluate(model, val_x, val_y, criterion, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_real_logit_mean": val_metrics["real_logit_mean"],
            "val_generated_logit_mean": val_metrics["generated_logit_mean"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}: train_loss={epoch_metrics['train_loss']:.6f} "
            f"val_loss={epoch_metrics['val_loss']:.6f} "
            f"val_acc={epoch_metrics['val_accuracy']:.4f}"
        )

    final_val = evaluate(model, val_x, val_y, criterion, device)
    output_dir = make_output_dir(project_root, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    histogram_path = output_dir / "logit_histogram.png"
    save_logit_histogram(final_val, histogram_path)

    metrics = {
        "real_embedding_path": str(real_embedding_path),
        "generated_embedding_path": str(generated_embedding_path),
        "num_per_class": int(split["num_per_class"]),
        "train_per_class": int(split["train_per_class"]),
        "val_per_class": int(split["val_per_class"]),
        "embedding_dim": int(classifier_cfg.embedding_dim),
        "epochs": int(classifier_cfg.epochs),
        "lr": float(classifier_cfg.lr),
        "weight_decay": float(classifier_cfg.weight_decay),
        "val_loss": final_val["loss"],
        "val_accuracy": final_val["accuracy"],
        "val_real_logit_mean": final_val["real_logit_mean"],
        "val_generated_logit_mean": final_val["generated_logit_mean"],
        "history": history,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifact_path = output_dir / "classifier.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classifier_type": "linear",
            "embedding_dim": int(classifier_cfg.embedding_dim),
            "feature_extractor": str(real_payload.get("feature_extractor", cfg.ratio.feature_extractor)),
            "feature_layer": str(real_payload.get("feature_layer", cfg.ratio.feature_layer)),
            "standardization": {
                "mean": train_mean.detach().cpu(),
                "std": train_std.detach().cpu(),
            },
            "real_embedding_path": str(real_embedding_path),
            "generated_embedding_path": str(generated_embedding_path),
            "metrics": metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        artifact_path,
    )

    print(f"Saved classifier artifact to: {artifact_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved logit histogram to: {histogram_path}")
    print(
        "Validation logit means: "
        f"real={metrics['val_real_logit_mean']:.6f}, "
        f"generated={metrics['val_generated_logit_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
