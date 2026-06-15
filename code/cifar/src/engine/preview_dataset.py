from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from src.utils import timestamped_output_path, unpack_batch


def take_samples(loader, num_samples: int) -> torch.Tensor:
    batches = []
    remaining = int(num_samples)
    for batch in loader:
        x = unpack_batch(batch)
        batches.append(x[:remaining])
        remaining -= x.shape[0]
        if remaining <= 0:
            break
    if not batches:
        raise ValueError("No samples available from the selected CIFAR split.")
    return torch.cat(batches, dim=0)


def make_output_path(cfg: DictConfig, project_root: Path, split: str) -> Path:
    return timestamped_output_path(
        output_dir=project_root / str(cfg.sampling.preview_dir),
        output_name=cfg.sampling.output_name,
        default_stem=f"{cfg.dataset.name}_{split}_real_samples",
        extension=".png",
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    split = str(cfg.dataset.preview_split)
    if split not in ("train", "test"):
        raise ValueError("dataset.preview_split must be 'train' or 'test'.")

    dataset_cfg = cfg.dataset.train if split == "train" else cfg.dataset.test
    dataset = instantiate(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.sampling.preview_num_samples),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    samples = take_samples(loader, int(cfg.sampling.preview_num_samples))
    preview = (samples.clamp(-1, 1) + 1) / 2
    grid = make_grid(preview, nrow=int(cfg.sampling.preview_nrow))

    output_path = make_output_path(cfg, project_root, split)
    save_image(grid, output_path)

    print(f"dataset={cfg.dataset.name} split={split} samples={samples.shape[0]}")
    print(f"saved real CIFAR preview to: {output_path}")


if __name__ == "__main__":
    main()
