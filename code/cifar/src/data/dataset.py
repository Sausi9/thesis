from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


Split = Literal["train", "test"]


@dataclass(frozen=True)
class DatasetBundle:
    train: DataLoader
    test: DataLoader
    shape: tuple[int, int, int]
    channels: int
    image_size: int
    name: str


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        *,
        root: str,
        split: Split,
        download: bool = True,
        augment: bool = False,
        return_labels: bool = False,
        max_samples: int | None = None,
    ):
        self.name = "cifar10"
        self.split = split
        self.return_labels = bool(return_labels)
        self.channels = 3
        self.image_size = 32
        self.shape = (self.channels, self.image_size, self.image_size)

        transform_steps = []
        if augment:
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        transform = transforms.Compose(transform_steps)

        dataset = datasets.CIFAR10(
            root=root,
            train=split == "train",
            download=bool(download),
            transform=transform,
        )

        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError(f"max_samples must be positive, got {max_samples}.")
            dataset = Subset(dataset, range(min(max_samples, len(dataset))))

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        if self.return_labels:
            return image, torch.tensor(label, dtype=torch.long)
        return image


def build_dataloaders(dataset: DictConfig, dataloader: DictConfig) -> DatasetBundle:
    train_dataset = instantiate(dataset.train)
    test_dataset = instantiate(dataset.test)

    if train_dataset.shape != test_dataset.shape:
        raise ValueError(
            f"Train/test shapes differ: {train_dataset.shape} != {test_dataset.shape}."
        )

    train_generator = torch.Generator().manual_seed(int(dataloader.train_seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(dataloader.train_batch_size),
        shuffle=bool(dataloader.shuffle_train),
        generator=train_generator,
        num_workers=int(dataloader.num_workers),
        pin_memory=bool(dataloader.pin_memory),
        drop_last=bool(dataloader.drop_last),
        persistent_workers=bool(dataloader.persistent_workers)
        if int(dataloader.num_workers) > 0
        else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(dataloader.test_batch_size),
        shuffle=False,
        num_workers=int(dataloader.num_workers),
        pin_memory=bool(dataloader.pin_memory),
        drop_last=False,
        persistent_workers=bool(dataloader.persistent_workers)
        if int(dataloader.num_workers) > 0
        else False,
    )

    return DatasetBundle(
        train=train_loader,
        test=test_loader,
        shape=tuple(train_dataset.shape),
        channels=int(train_dataset.channels),
        image_size=int(train_dataset.image_size),
        name=str(dataset.name),
    )
