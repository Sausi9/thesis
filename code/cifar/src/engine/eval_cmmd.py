from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

cache_root = Path(tempfile.gettempdir()) / "cifar_cmmd_eval_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import CIFAR10Dataset
from src.metrics.cmmd import CMMDClipEncoder, cmmd, extract_clip_embeddings
from src.ratio.inception import ImageTensorDataset
from src.utils import resolve_device, resolve_path


def load_samples(path: Path, num_samples: int) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu")
    samples = payload.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 4:
        raise ValueError(f"Sample payload has no valid samples tensor: {path}.")
    if samples.shape[0] < int(num_samples):
        raise ValueError(
            f"Requested {num_samples} samples from {path}, but it contains {samples.shape[0]}."
        )
    return samples[: int(num_samples)].detach().cpu(), payload


def cache_path_for(cache_dir: Path, source_id: str) -> Path:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.pt"


def load_or_extract_embeddings(
    *,
    cache_dir: Path,
    source_id: str,
    metadata: dict,
    encoder: CMMDClipEncoder,
    dataloader: DataLoader,
    num_samples: int,
) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for(cache_dir, source_id)
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("metadata") == metadata:
            features = cached.get("features")
            if torch.is_tensor(features) and features.shape[0] == int(num_samples):
                print(f"Loaded cached CMMD embeddings from: {cache_path}")
                return features.float()

    features = extract_clip_embeddings(
        encoder=encoder,
        dataloader=dataloader,
        max_samples=int(num_samples),
    )
    torch.save({"features": features, "metadata": metadata}, cache_path)
    print(f"Saved CMMD embeddings to: {cache_path}")
    return features


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    if cfg.cmmd.unguided_sample_path is None or cfg.cmmd.guided_sample_path is None:
        raise ValueError(
            "Set cmmd.unguided_sample_path and cmmd.guided_sample_path."
        )

    unguided_path = resolve_path(project_root, str(cfg.cmmd.unguided_sample_path))
    guided_path = resolve_path(project_root, str(cfg.cmmd.guided_sample_path))
    num_samples = int(cfg.cmmd.num_samples)
    if num_samples <= 1:
        raise ValueError("cmmd.num_samples must be greater than one.")
    unguided, unguided_payload = load_samples(unguided_path, num_samples)
    guided, guided_payload = load_samples(guided_path, num_samples)

    model_cache_dir = (
        None
        if cfg.cmmd.model_cache_dir is None
        else resolve_path(project_root, str(cfg.cmmd.model_cache_dir))
    )
    encoder = CMMDClipEncoder(
        device=device,
        model_name=str(cfg.cmmd.model_name),
        pretrained=str(cfg.cmmd.pretrained),
        cache_dir=model_cache_dir,
    )

    loader_kwargs = {
        "batch_size": int(cfg.cmmd.embedding_batch_size),
        "shuffle": False,
        "num_workers": int(cfg.cmmd.num_workers),
        "pin_memory": bool(cfg.cmmd.pin_memory),
    }
    real_dataset = CIFAR10Dataset(
        root=str(resolve_path(project_root, str(cfg.dataset.root))),
        split=str(cfg.cmmd.reference_split),
        download=bool(cfg.dataset.download),
        augment=False,
        return_labels=False,
        max_samples=num_samples,
    )
    cache_dir = resolve_path(project_root, str(cfg.cmmd.embedding_cache_dir))
    shared_metadata = {
        "num_samples": num_samples,
        "model_name": str(cfg.cmmd.model_name),
        "pretrained": str(cfg.cmmd.pretrained),
        "normalized": True,
        "preprocessing": "resize_bicubic_clip_normalize",
    }
    real_metadata = {
        **shared_metadata,
        "source": "cifar10",
        "split": str(cfg.cmmd.reference_split),
    }
    real_features = load_or_extract_embeddings(
        cache_dir=cache_dir,
        source_id=json.dumps(real_metadata, sort_keys=True),
        metadata=real_metadata,
        encoder=encoder,
        dataloader=DataLoader(real_dataset, **loader_kwargs),
        num_samples=num_samples,
    )

    def sample_features(path: Path, samples: torch.Tensor, sample_type: str):
        stat = path.stat()
        metadata = {
            **shared_metadata,
            "source": str(path),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "sample_type": sample_type,
        }
        return load_or_extract_embeddings(
            cache_dir=cache_dir,
            source_id=json.dumps(metadata, sort_keys=True),
            metadata=metadata,
            encoder=encoder,
            dataloader=DataLoader(ImageTensorDataset(samples), **loader_kwargs),
            num_samples=num_samples,
        )

    unguided_features = sample_features(
        unguided_path,
        unguided,
        str(unguided_payload.get("sample_type", "unknown")),
    )
    guided_features = sample_features(
        guided_path,
        guided,
        str(guided_payload.get("sample_type", "unknown")),
    )

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cmmd_kwargs = {
        "device": device,
        "sigma": float(cfg.cmmd.sigma),
        "scale": float(cfg.cmmd.scale),
        "batch_size": int(cfg.cmmd.kernel_batch_size),
    }
    unguided_cmmd = cmmd(real_features, unguided_features, **cmmd_kwargs)
    guided_cmmd = cmmd(real_features, guided_features, **cmmd_kwargs)
    relative_change = (
        float("nan")
        if unguided_cmmd == 0.0
        else (guided_cmmd - unguided_cmmd) / unguided_cmmd
    )

    default_name = f"{unguided_path.stem}_vs_{guided_path.stem}_{cfg.cmmd.reference_split}"
    output_name = default_name if cfg.cmmd.output_name is None else str(cfg.cmmd.output_name)
    output_dir = resolve_path(project_root, str(cfg.cmmd.output_dir)) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    values = [unguided_cmmd, guided_cmmd]
    bars = ax.bar(["unguided", "guided TDS"], values, color=["#6b7280", "#2563eb"])
    ax.set_ylabel("CMMD (lower is better)")
    ax.set_title(f"CMMD against CIFAR-10 {cfg.cmmd.reference_split}")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    fig.savefig(output_dir / "cmmd_comparison.png", dpi=200)
    plt.close(fig)

    metrics = {
        "reference": f"cifar10-{cfg.cmmd.reference_split}",
        "num_samples_per_set": num_samples,
        "unguided_sample_file": str(unguided_path),
        "guided_sample_file": str(guided_path),
        "unguided_sample_type": str(unguided_payload.get("sample_type", "unknown")),
        "guided_sample_type": str(guided_payload.get("sample_type", "unknown")),
        "model_name": str(cfg.cmmd.model_name),
        "pretrained": str(cfg.cmmd.pretrained),
        "normalized_embeddings": True,
        "estimator": "biased_gaussian_kernel_mmd",
        "sigma": float(cfg.cmmd.sigma),
        "scale": float(cfg.cmmd.scale),
        "unguided_cmmd": unguided_cmmd,
        "guided_cmmd": guided_cmmd,
        "cmmd_change": guided_cmmd - unguided_cmmd,
        "cmmd_relative_change": relative_change,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved CMMD evaluation to: {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(metrics), resolve=True))


if __name__ == "__main__":
    main()
