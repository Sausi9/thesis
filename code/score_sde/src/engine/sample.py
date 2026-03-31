import argparse
from pathlib import Path

import torch
import yaml

from src.data.datasets import DATASET_SPECS
from src.engine.artifacts import resolve_artifact_path
from src.models.unet import UNet


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config structure in {config_path}.")
    return config


def resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample images from a trained Score-SDE model."
    )
    parser.add_argument(
        "dataset",
        choices=tuple(sorted(DATASET_SPECS)),
        help="Dataset family to sample from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "sample.yaml")

    dataset_name = args.dataset.lower()
    dataset_spec = DATASET_SPECS[dataset_name]
    device = resolve_device(config.get("device", "auto"))
    artifact_dir = project_root / "artifacts" / "models"

    artifact_name = config.get("artifact_name")
    artifact_path = resolve_artifact_path(
        artifact_dir=artifact_dir,
        dataset_name=dataset_name,
        artifact_name=artifact_name,
    )

    checkpoint = torch.load(artifact_path, map_location=device)
    model_cfg = checkpoint["model_config"]
    if int(model_cfg["in_channels"]) != int(dataset_spec["channels"]):
        raise ValueError(
            f"Artifact '{artifact_path.name}' does not match dataset '{dataset_name}'."
        )

    model = UNet(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        model_channels=int(model_cfg["model_channels"]),
        channel_mult=tuple(model_cfg["channel_mult"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        attn_resolutions=tuple(model_cfg["attn_resolutions"]),
        dropout=float(model_cfg["dropout"]),
        resamp_with_conv=bool(model_cfg["resamp_with_conv"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    raise NotImplementedError(
        "Score-SDE sampling is not implemented yet. "
        "The artifact loading scaffold is ready for predictor-corrector or ODE sampling."
    )


if __name__ == "__main__":
    main()
