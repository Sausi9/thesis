import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

from src.data.datasets import get_dataset
from src.diffusion.process import Diffusion
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
    parser = argparse.ArgumentParser(description="Train the DDPM model.")
    parser.add_argument(
        "dataset",
        choices=("mnist", "cifar"),
        help="Dataset to train on.",
    )
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
        train_cfg.get("test_batch_size", train_cfg.get("eval_batch_size", train_batch_size))
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
    model.train()

    learning_rate = float(train_cfg["learning_rate"])
    num_epochs = int(train_cfg["num_epochs"])
    checkpoint_every = int(train_cfg.get("checkpoint_every", 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    diff_cfg = config["diffusion"]
    T = int(diff_cfg["T"])
    beta_start = float(diff_cfg["beta_start"])
    beta_end = float(diff_cfg["beta_end"])
    betas = torch.linspace(beta_start, beta_end, T)
    diffusion = Diffusion(betas=betas, T=T)

    run_stem = str(config.get("run_name") or dataset_spec["default_run_name"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_stem}_{run_id}"
    checkpoint_dir = project_root / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = project_root / "artifacts" / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config["dataset"] = dataset_name
    config["model"] = model_cfg
    with (checkpoint_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    best_loss = float("inf")
    best_model_state = None
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for x0, _ in pbar:
            x0 = x0.to(device)
            B = x0.size(0)
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

            loss = diffusion.train(model, optimizer, x0, t)
            loss_val = loss.item()
            epoch_loss += loss_val
            num_steps += 1
            global_step += 1

            pbar.set_postfix(
                step_loss=f"{loss_val:.4f}", epoch_avg=f"{epoch_loss / num_steps:.4f}"
            )

        epoch_avg = epoch_loss / num_steps
        print(f"Epoch {epoch + 1}: avg_loss={epoch_avg:.4f}")

        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": {
                "epoch_avg_loss": epoch_avg,
                "best_loss": min(best_loss, epoch_avg),
            },
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")

        if epoch_avg < best_loss:
            best_loss = epoch_avg
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

    final_model_artifact = {
        "run_name": run_name,
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "model_config": model_cfg,
        "diffusion_config": diff_cfg,
    }
    torch.save(final_model_artifact, artifact_dir / f"{run_name}_final.pt")

    if best_model_state is not None:
        best_model_artifact = {
            "run_name": run_name,
            "model_state_dict": best_model_state,
            "model_config": model_cfg,
            "diffusion_config": diff_cfg,
            "metrics": {
                "best_loss": best_loss,
            },
        }
        torch.save(best_model_artifact, artifact_dir / f"{run_name}_best.pt")

    print(f"Saved checkpoints to: {checkpoint_dir}")
    print(f"Saved model artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
