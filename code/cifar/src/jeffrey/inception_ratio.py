from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.ratio.classifier import LinearRatioClassifier, standardize


class DifferentiableInception2048(nn.Module):
    """FID-compatible Inception features with continuous, differentiable inputs."""

    def __init__(
        self,
        *,
        feature_extractor: str = "inception-v3-compat",
        feature_layer: str = "2048",
        preprocessing: str = "continuous",
        verbose: bool = False,
    ):
        super().__init__()
        if feature_extractor != "inception-v3-compat":
            raise ValueError(
                "Only torch-fidelity's inception-v3-compat extractor is supported, "
                f"got {feature_extractor!r}."
            )
        if str(feature_layer) != "2048":
            raise ValueError(f"Only feature_layer='2048' is supported, got {feature_layer!r}.")
        if preprocessing != "continuous":
            raise ValueError(f"Only preprocessing='continuous' is supported, got {preprocessing!r}.")

        from torch_fidelity.feature_extractor_inceptionv3 import (
            FeatureExtractorInceptionV3,
            interpolate_bilinear_2d_like_tensorflow1x,
        )

        self.extractor = FeatureExtractorInceptionV3(
            feature_extractor,
            [str(feature_layer)],
            verbose=bool(verbose),
        )
        self.interpolate = interpolate_bilinear_2d_like_tensorflow1x
        self.feature_extractor = feature_extractor
        self.feature_layer = str(feature_layer)
        self.preprocessing = preprocessing

        self.extractor.eval()
        self.extractor.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected images [B, C, H, W], got {tuple(images.shape)}.")
        if images.shape[1:] != (3, 32, 32):
            raise ValueError(f"Expected CIFAR images [B, 3, 32, 32], got {tuple(images.shape)}.")

        x = ((images.clamp(-1, 1) + 1.0) * 127.5).to(
            dtype=self.extractor.feature_extractor_internal_dtype
        )
        x = self.interpolate(
            x,
            size=(self.extractor.INPUT_IMAGE_SIZE, self.extractor.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
        x = (x - 128.0) / 128.0

        extractor = self.extractor
        x = extractor.Conv2d_1a_3x3(x)
        x = extractor.Conv2d_2a_3x3(x)
        x = extractor.Conv2d_2b_3x3(x)
        x = extractor.MaxPool_1(x)
        x = extractor.Conv2d_3b_1x1(x)
        x = extractor.Conv2d_4a_3x3(x)
        x = extractor.MaxPool_2(x)
        x = extractor.Mixed_5b(x)
        x = extractor.Mixed_5c(x)
        x = extractor.Mixed_5d(x)
        x = extractor.Mixed_6a(x)
        x = extractor.Mixed_6b(x)
        x = extractor.Mixed_6c(x)
        x = extractor.Mixed_6d(x)
        x = extractor.Mixed_6e(x)
        x = extractor.Mixed_7a(x)
        x = extractor.Mixed_7b(x)
        x = extractor.Mixed_7c(x)
        x = extractor.AvgPool(x)
        return torch.flatten(x, 1).to(torch.float32)


class InceptionRatioPotential(nn.Module):
    """Differentiable log-ratio potential from a saved linear ratio classifier."""

    def __init__(
        self,
        *,
        classifier_path: str | Path,
        feature_extractor: str = "inception-v3-compat",
        feature_layer: str = "2048",
        preprocessing: str = "continuous",
        logit_clip: float | None = 20.0,
        map_location: str | torch.device = "cpu",
    ):
        super().__init__()
        self.classifier_path = str(classifier_path)
        self.logit_clip = None if logit_clip is None else float(logit_clip)
        if self.logit_clip is not None and self.logit_clip <= 0.0:
            raise ValueError(f"logit_clip must be positive or null, got {logit_clip}.")

        artifact = torch.load(classifier_path, map_location=map_location)
        embedding_dim = int(artifact.get("embedding_dim", 2048))
        if embedding_dim != 2048:
            raise ValueError(f"Expected embedding_dim=2048, got {embedding_dim}.")

        standardization = artifact.get("standardization")
        if not isinstance(standardization, dict):
            raise KeyError("Classifier artifact must contain standardization mean/std.")
        mean = standardization.get("mean")
        std = standardization.get("std")
        if not torch.is_tensor(mean) or not torch.is_tensor(std):
            raise TypeError("Classifier artifact standardization mean/std must be tensors.")

        self.inception = DifferentiableInception2048(
            feature_extractor=feature_extractor,
            feature_layer=feature_layer,
            preprocessing=preprocessing,
        )
        self.classifier = LinearRatioClassifier(embedding_dim)
        self.classifier.load_state_dict(artifact["model_state_dict"])
        self.classifier.eval()
        self.classifier.requires_grad_(False)

        self.register_buffer("feature_mean", mean.detach().float())
        self.register_buffer("feature_std", std.detach().float().clamp_min(1e-6))

        self.feature_extractor = str(feature_extractor)
        self.feature_layer = str(feature_layer)
        self.preprocessing = str(preprocessing)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.inception(images)
        normalized = standardize(features, self.feature_mean, self.feature_std)
        logits = self.classifier(normalized)
        if self.logit_clip is not None:
            logits = logits.clamp(-self.logit_clip, self.logit_clip)
        return logits


@torch.no_grad()
def compute_ratio_logits(
    *,
    potential: InceptionRatioPotential,
    samples: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    potential.eval()
    logits = []
    for start in range(0, int(samples.shape[0]), int(batch_size)):
        batch = samples[start : start + int(batch_size)].to(device)
        logits.append(potential(batch).detach().cpu())
    return torch.cat(logits, dim=0)
