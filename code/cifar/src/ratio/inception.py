from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ImageTensorDataset(Dataset):
    """Dataset adapter for image tensors stored in model space [-1, 1]."""

    def __init__(self, images: torch.Tensor):
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [N, C, H, W], got {tuple(images.shape)}.")
        if images.shape[1:] != (3, 32, 32):
            raise ValueError(f"Expected CIFAR-shaped images [N, 3, 32, 32], got {tuple(images.shape)}.")
        self.images = images.detach().cpu()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]


def images_to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert CIFAR tensors from [-1, 1] float space to [0, 255] uint8."""

    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}.")
    images = ((images.clamp(-1, 1) + 1) * 127.5).round()
    return images.clamp(0, 255).to(torch.uint8)


def build_inception_extractor(
    *,
    feature_extractor: str,
    feature_layer: str,
    device: torch.device,
    verbose: bool = False,
):
    if feature_extractor != "inception-v3-compat":
        raise ValueError(
            "Only torch-fidelity's inception-v3-compat feature extractor is supported, "
            f"got {feature_extractor!r}."
        )

    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    extractor = FeatureExtractorInceptionV3(
        feature_extractor,
        [str(feature_layer)],
        verbose=bool(verbose),
    )
    extractor.to(device)
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor


@torch.no_grad()
def extract_features(
    *,
    extractor,
    dataloader,
    device: torch.device,
    max_samples: int | None = None,
) -> torch.Tensor:
    features = []
    seen = 0

    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        if max_samples is not None:
            remaining = int(max_samples) - seen
            if remaining <= 0:
                break
            batch = batch[:remaining]

        batch = images_to_uint8(batch).to(device, non_blocking=True)
        extracted = extractor(batch)
        if isinstance(extracted, tuple):
            extracted = extracted[0]
        features.append(extracted.detach().cpu().float())
        seen += int(batch.shape[0])

    if not features:
        raise ValueError("No images were available for Inception feature extraction.")

    return torch.cat(features, dim=0)
