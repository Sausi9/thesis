from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


Split = Literal["train", "test", "validation"]


@dataclass(frozen=True)
class DatasetBundle:
    train: DataLoader
    test: DataLoader
    dim: int
    name: str


class BivariateGaussianDataset(Dataset):
    """In-memory samples from a configurable 2D Gaussian distribution."""

    def __init__(
        self,
        *,
        mean: list[float],
        covariance: list[list[float]],
        num_samples: int,
        seed: int,
        split: Split,
        return_labels: bool = False,
    ):
        self.name = "bivariate_gaussian"
        self.split = split
        self.return_labels = bool(return_labels)
        self.mean, self.covariance = _validate_gaussian_parameters(mean, covariance)
        self.num_samples = _validate_positive_int(num_samples, "num_samples")
        self.seed = int(seed)
        self.samples = _sample_gaussian(
            num_samples=self.num_samples,
            mean=self.mean,
            covariance=self.covariance,
            seed=self.seed,
        )

    @property
    def dim(self) -> int:
        return int(self.mean.numel())

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if not self.return_labels:
            return sample
        return sample, torch.tensor(0, dtype=torch.long)


def build_dataloaders(dataset: DictConfig, dataloader: DictConfig) -> DatasetBundle:
    """Instantiate Hydra-configured datasets and wrap them in DataLoaders."""
    train_dataset = instantiate(dataset.train)
    test_dataset = instantiate(dataset.test)

    if train_dataset.dim != test_dataset.dim:
        raise ValueError(
            f"Train/test dimensions differ: {train_dataset.dim} != {test_dataset.dim}."
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
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(dataloader.test_batch_size),
        shuffle=False,
        num_workers=int(dataloader.num_workers),
        pin_memory=bool(dataloader.pin_memory),
        drop_last=False,
    )

    return DatasetBundle(
        train=train_loader,
        test=test_loader,
        dim=train_dataset.dim,
        name=str(dataset.name),
    )


def _validate_gaussian_parameters(
    mean: list[float],
    covariance: list[list[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    covariance_tensor = torch.tensor(covariance, dtype=torch.float32)

    if mean_tensor.ndim != 1:
        raise ValueError("Gaussian mean must be a 1D list.")
    if mean_tensor.numel() != 2:
        raise ValueError("Bivariate Gaussian mean must contain exactly 2 values.")
    if covariance_tensor.ndim != 2:
        raise ValueError("Gaussian covariance must be a 2D list.")
    if covariance_tensor.shape != (2, 2):
        raise ValueError("Bivariate Gaussian covariance must be a 2x2 matrix.")
    if not torch.allclose(covariance_tensor, covariance_tensor.T):
        raise ValueError("Gaussian covariance must be symmetric.")

    eigenvalues = torch.linalg.eigvalsh(covariance_tensor)
    if torch.any(eigenvalues <= 0):
        raise ValueError("Gaussian covariance must be positive definite.")

    return mean_tensor, covariance_tensor


def _validate_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _sample_gaussian(
    *,
    num_samples: int,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    chol = torch.linalg.cholesky(covariance)
    noise = torch.randn(num_samples, mean.numel(), generator=generator)
    return mean + noise @ chol.T
