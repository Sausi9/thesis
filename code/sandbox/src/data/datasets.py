from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


DATASET_SPECS = {
    "mnist": {
        "loader": "get_mnist",
        "channels": 1,
        "image_size": 28,
        "default_run_name": "ddpm_mnist",
    },
    "cifar": {
        "loader": "get_cifar10",
        "channels": 3,
        "image_size": 32,
        "default_run_name": "ddpm_cifar",
    },
}


def get_mnist(train_batch_size: int = 64, test_batch_size: int = 1000):
    # 1. Define Transforms (Critical for performance)
    # MNIST images are 28x28. ToTensor() scales pixels from [0, 255] to [0.0, 1.0].
    # Normalize uses the Mean (0.1307) and Std Dev (0.3081) of the MNIST dataset.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # 2. Download/Load the Datasets
    mnist_train = datasets.MNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar10(train_batch_size: int = 64, test_batch_size: int = 64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar_train = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )

    cifar_test = datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        cifar_train, batch_size=train_batch_size, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        cifar_test, batch_size=test_batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


def get_dataset(
    dataset_name: str, train_batch_size: int | None = None, test_batch_size: int | None = None
):
    dataset_key = dataset_name.lower()
    try:
        spec = DATASET_SPECS[dataset_key]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Expected one of: {valid}."
        ) from exc

    loader = globals()[spec["loader"]]
    loader_kwargs = {}
    if train_batch_size is not None:
        loader_kwargs["train_batch_size"] = train_batch_size
    if test_batch_size is not None:
        loader_kwargs["test_batch_size"] = test_batch_size
    train_loader, test_loader = loader(**loader_kwargs)
    return train_loader, test_loader, spec.copy()
