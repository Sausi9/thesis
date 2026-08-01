from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

cache_root = Path(tempfile.gettempdir()) / "cifar_paired_inception_eval_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.engine.inception_tds_paired_sample import display_space, pixel_distances
from src.ratio.inception import ImageTensorDataset, build_inception_extractor, extract_features
from src.utils import resolve_device, resolve_path


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([x.float(), y.float()]))[0, 1])


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty(values.numel(), dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks


def save_side_by_side_pairs(
    *,
    guided: torch.Tensor,
    unguided: torch.Tensor,
    distances: torch.Tensor,
    indices: torch.Tensor,
    pairs_per_row: int,
    output_path: Path,
) -> None:
    count = int(indices.numel())
    columns = 2 * min(int(pairs_per_row), count)
    rows = math.ceil(count / int(pairs_per_row))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(1.9 * columns, 2.4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis in axes.flat:
        axis.axis("off")

    guided_display = display_space(guided[indices]).permute(0, 2, 3, 1).numpy()
    unguided_display = display_space(unguided[indices]).permute(0, 2, 3, 1).numpy()
    for rank, pair_index in enumerate(indices.tolist()):
        row = rank // int(pairs_per_row)
        pair_column = rank % int(pairs_per_row)
        left = axes[row, pair_column * 2]
        right = axes[row, pair_column * 2 + 1]
        left.imshow(unguided_display[rank])
        right.imshow(guided_display[rank])
        left.set_title(f"rank {rank + 1}\nunguided", fontsize=9)
        right.set_title(
            f"guided\nInception L2={float(distances[pair_index]):.3f}",
            fontsize=9,
        )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    if cfg.sampling.sample_path is None:
        raise ValueError("Set sampling.sample_path to a completed paired sample payload.")

    sample_path = resolve_path(project_root, str(cfg.sampling.sample_path))
    payload = torch.load(sample_path, map_location="cpu")
    if not bool(payload.get("complete", True)):
        raise ValueError(f"Paired payload is incomplete: {sample_path}.")
    if "guided_samples" not in payload or "unguided_samples" not in payload:
        raise KeyError("Payload must contain guided_samples and unguided_samples.")

    guided = payload["guided_samples"].detach().cpu()
    unguided = payload["unguided_samples"].detach().cpu()
    if guided.shape != unguided.shape or guided.ndim != 4:
        raise ValueError(
            f"Expected matching paired tensors [N,C,H,W], got {guided.shape} and {unguided.shape}."
        )

    top_k = int(cfg.paired.top_k)
    if top_k <= 0 or top_k > guided.shape[0]:
        raise ValueError("paired.top_k must be between 1 and the number of pairs.")
    if int(cfg.paired.pairs_per_row) <= 0:
        raise ValueError("paired.pairs_per_row must be positive.")

    extractor = build_inception_extractor(
        feature_extractor=str(cfg.ratio.feature_extractor),
        feature_layer=str(cfg.ratio.feature_layer),
        device=device,
        verbose=False,
    )
    loader_kwargs = {
        "batch_size": int(cfg.ratio.batch_size),
        "shuffle": False,
        "num_workers": int(cfg.ratio.num_workers),
        "pin_memory": bool(cfg.ratio.pin_memory),
    }
    guided_features = extract_features(
        extractor=extractor,
        dataloader=DataLoader(ImageTensorDataset(guided), **loader_kwargs),
        device=device,
    )
    unguided_features = extract_features(
        extractor=extractor,
        dataloader=DataLoader(ImageTensorDataset(unguided), **loader_kwargs),
        device=device,
    )

    inception_l2 = (guided_features - unguided_features).norm(dim=1)
    pixel_l2, pixel_mse = pixel_distances(guided, unguided)
    ranking = torch.argsort(inception_l2, descending=True)
    top_indices = ranking[:top_k]

    output_dir = resolve_path(project_root, str(cfg.paired.output_dir)) / sample_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    save_side_by_side_pairs(
        guided=guided,
        unguided=unguided,
        distances=inception_l2,
        indices=top_indices,
        pairs_per_row=int(cfg.paired.pairs_per_row),
        output_path=output_dir / f"top{top_k}_inception_l2_pairs.png",
    )

    pearson = _correlation(pixel_l2, inception_l2)
    spearman = _correlation(_ranks(pixel_l2), _ranks(inception_l2))
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.scatter(pixel_l2.numpy(), inception_l2.numpy(), s=20, alpha=0.65)
    ax.set_xlabel("pixel-space L2 distance")
    ax.set_ylabel("Inception-embedding L2 distance")
    ax.set_title(f"Paired distances (Pearson r={pearson:.3f})")
    fig.savefig(output_dir / "inception_vs_pixel_distance.png", dpi=200)
    plt.close(fig)

    with (output_dir / "inception_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "pair_index", "inception_l2", "pixel_l2", "pixel_mse"]
        )
        for rank, pair_index in enumerate(ranking.tolist(), start=1):
            writer.writerow(
                [
                    rank,
                    pair_index,
                    float(inception_l2[pair_index]),
                    float(pixel_l2[pair_index]),
                    float(pixel_mse[pair_index]),
                ]
            )

    metrics = {
        "sample_file": str(sample_path),
        "sample_type": str(payload.get("sample_type", "unknown")),
        "pairing_semantics": str(payload.get("pairing_semantics", "unknown")),
        "num_pairs": int(guided.shape[0]),
        "feature_extractor": str(cfg.ratio.feature_extractor),
        "feature_layer": str(cfg.ratio.feature_layer),
        "preprocessing": "fid_compatible_uint8",
        "inception_l2_mean": float(inception_l2.mean()),
        "inception_l2_std": float(
            inception_l2.std(unbiased=inception_l2.numel() > 1)
        ),
        "inception_l2_min": float(inception_l2.min()),
        "inception_l2_max": float(inception_l2.max()),
        "pixel_inception_pearson": pearson,
        "pixel_inception_spearman": spearman,
        "top_pair_indices": [int(value) for value in top_indices.tolist()],
        "top_pair_inception_l2": [
            float(inception_l2[index]) for index in top_indices.tolist()
        ],
    }
    with (output_dir / "inception_distance_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Loaded paired samples from: {sample_path}")
    print(f"Saved paired Inception analysis to: {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(metrics), resolve=True))


if __name__ == "__main__":
    main()
