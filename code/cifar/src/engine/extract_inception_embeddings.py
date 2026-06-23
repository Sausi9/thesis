from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import CIFAR10Dataset
from src.ratio.inception import ImageTensorDataset, build_inception_extractor, extract_features
from src.utils import find_latest_sample, resolve_device, resolve_path


def make_embedding_path(output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{stem}_{timestamp}.pt"


def load_generated_samples(project_root: Path, cfg: DictConfig) -> tuple[torch.Tensor, Path, dict]:
    samples_dir = resolve_path(project_root, str(cfg.sampling.output_dir))
    if cfg.ratio.generated_sample_path is None:
        sample_path = find_latest_sample(samples_dir, sample_type="model")
    else:
        sample_path = resolve_path(project_root, str(cfg.ratio.generated_sample_path))

    payload = torch.load(sample_path, map_location="cpu")
    if payload.get("sample_type") != "model":
        raise ValueError(
            "Ratio classifier expects unconditional model samples, "
            f"got sample_type={payload.get('sample_type')!r} from {sample_path}."
        )
    samples = payload["samples"]
    max_samples = cfg.ratio.generated_max_samples
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    return samples, sample_path, payload


def save_embedding_artifact(
    *,
    path: Path,
    features: torch.Tensor,
    source_kind: str,
    source: dict,
    cfg: DictConfig,
) -> None:
    artifact = {
        "features": features.detach().cpu().float(),
        "source_kind": source_kind,
        "source": source,
        "num_samples": int(features.shape[0]),
        "embedding_dim": int(features.shape[1]),
        "feature_extractor": str(cfg.ratio.feature_extractor),
        "feature_layer": str(cfg.ratio.feature_layer),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(artifact, path)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))

    extractor = build_inception_extractor(
        feature_extractor=str(cfg.ratio.feature_extractor),
        feature_layer=str(cfg.ratio.feature_layer),
        device=device,
        verbose=False,
    )

    real_max_samples = cfg.ratio.real_max_samples
    real_dataset = CIFAR10Dataset(
        root=str(resolve_path(project_root, str(cfg.ratio.datasets_root))),
        split=str(cfg.ratio.real_split),
        download=bool(cfg.ratio.datasets_download),
        augment=False,
        return_labels=False,
        max_samples=int(real_max_samples) if real_max_samples is not None else None,
    )
    real_loader = DataLoader(
        real_dataset,
        batch_size=int(cfg.ratio.batch_size),
        shuffle=False,
        num_workers=int(cfg.ratio.num_workers),
        pin_memory=bool(cfg.ratio.pin_memory),
    )
    real_features = extract_features(
        extractor=extractor,
        dataloader=real_loader,
        device=device,
        max_samples=int(real_max_samples) if real_max_samples is not None else None,
    )

    generated_samples, generated_sample_path, generated_payload = load_generated_samples(
        project_root, cfg
    )
    generated_dataset = ImageTensorDataset(generated_samples)
    generated_loader = DataLoader(
        generated_dataset,
        batch_size=int(cfg.ratio.batch_size),
        shuffle=False,
        num_workers=int(cfg.ratio.num_workers),
        pin_memory=bool(cfg.ratio.pin_memory),
    )
    generated_features = extract_features(
        extractor=extractor,
        dataloader=generated_loader,
        device=device,
        max_samples=None,
    )

    output_dir = resolve_path(project_root, str(cfg.ratio.embedding_dir))
    layer = str(cfg.ratio.feature_layer)
    real_path = make_embedding_path(
        output_dir,
        f"cifar10_{cfg.ratio.real_split}_real_inception_{layer}",
    )
    generated_path = make_embedding_path(
        output_dir,
        f"{generated_sample_path.stem}_generated_inception_{layer}",
    )

    save_embedding_artifact(
        path=real_path,
        features=real_features,
        source_kind="real",
        source={
            "dataset": "cifar10",
            "split": str(cfg.ratio.real_split),
            "augment": False,
            "max_samples": None if real_max_samples is None else int(real_max_samples),
        },
        cfg=cfg,
    )
    save_embedding_artifact(
        path=generated_path,
        features=generated_features,
        source_kind="generated",
        source={
            "sample_path": str(generated_sample_path),
            "sample_type": str(generated_payload.get("sample_type", "unknown")),
            "artifact_path": str(generated_payload.get("artifact_path", "")),
            "max_samples": None
            if cfg.ratio.generated_max_samples is None
            else int(cfg.ratio.generated_max_samples),
        },
        cfg=cfg,
    )

    print(f"Saved real embeddings to: {real_path}")
    print(f"Real features shape: {tuple(real_features.shape)}")
    print(f"Saved generated embeddings to: {generated_path}")
    print(f"Generated features shape: {tuple(generated_features.shape)}")


if __name__ == "__main__":
    main()
