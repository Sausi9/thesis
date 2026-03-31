import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.data.datasets import get_dataset
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
    parser = argparse.ArgumentParser(description="Train the Score-SDE model.")
    parser.add_argument(
        "dataset",
        choices=("mnist", "cifar"),
        help="Dataset to train on.",
    )
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    dataset_name = args.dataset.lower()
    dataset_config = dataset_name + "_train.yaml"
    config = load_config(project_root / "configs" / dataset_config)
    train_cfg = config["train"]
    train_batch_size = int(train_cfg["batch_size"])
    test_batch_size = int(
        train_cfg.get(
            "test_batch_size", train_cfg.get("eval_batch_size", train_batch_size)
        )
    )
    train_loader, _, dataset_spec = get_dataset(
        dataset_name,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
    )

    device = resolve_device(config.get("device", "auto"))
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    model_cfg = dict(config["model"])
    model_cfg["in_channels"] = dataset_spec["channels"]
    model_cfg["out_channels"] = dataset_spec["channels"]
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(train_cfg["learning_rate"])
    )

    artifact_dir = project_root / "artifacts" / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        checkpoint_dir = resume_path.parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_name = checkpoint_dir.name
    else:
        run_stem = str(config.get("run_name") or dataset_spec["default_run_name"])
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_stem}_{run_id}"
        checkpoint_dir = project_root / "checkpoints" / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config["dataset"] = dataset_name
    config["model"] = model_cfg
    with (checkpoint_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    _ = train_loader, optimizer
    raise NotImplementedError(
        "Score-SDE training is not implemented yet. "
        "Reusable scaffolding is in place; next step is the continuous-time loss and sampler."
    )


if __name__ == "__main__":
    main()
