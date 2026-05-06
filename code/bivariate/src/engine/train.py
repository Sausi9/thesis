from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.data.dataset import build_dataloaders
from src.utils import make_run_name, resolve_device, resolve_path, save_yaml


def unpack_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def score_matching_loss(model, sde, x0: torch.Tensor) -> torch.Tensor:
    batch_size = x0.shape[0]
    t = (
        torch.rand(batch_size, device=x0.device)
        * (sde.config.t_max - sde.config.t_min)
        + sde.config.t_min
    )
    x_t, epsilon, std = sde.forward_noising(x0, t)
    score_pred = model(x_t, t)
    return ((score_pred * std + epsilon) ** 2).mean()


def resolve_checkpoint_path(project_root: Path, checkpoint_path: str) -> Path:
    return resolve_path(project_root, checkpoint_path)


def assert_resume_config_matches(cfg: DictConfig, checkpoint: dict) -> None:
    previous_cfg = checkpoint.get("config")
    if previous_cfg is None:
        return

    current = OmegaConf.to_container(cfg, resolve=True)
    keys_to_compare = ("dataset", "dataloader", "model", "sde", "optimizer")
    mismatches = [
        key
        for key in keys_to_compare
        if previous_cfg.get(key) != current.get(key)
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            "Resume config differs from checkpoint for: "
            f"{joined}. Re-run with resume.strict_config=false if this is intentional."
        )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    device = resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))

    data = build_dataloaders(cfg.dataset, cfg.dataloader)
    model = instantiate(cfg.model).to(device)
    sde = instantiate(cfg.sde)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    artifact_dir = project_root / str(cfg.training.artifacts_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = int(cfg.training.num_epochs)
    checkpoint_every = int(cfg.training.checkpoint_every)
    log_every = int(cfg.training.log_every)

    best_loss = float("inf")
    best_model_state = None
    start_epoch = 0
    global_step = 0

    if cfg.resume.checkpoint_path is not None:
        resume_path = resolve_checkpoint_path(project_root, str(cfg.resume.checkpoint_path))
        checkpoint_dir = resume_path.parent
        run_name = checkpoint_dir.name
        checkpoint = torch.load(resume_path, map_location=device)

        if bool(cfg.resume.strict_config):
            assert_resume_config_matches(cfg, checkpoint)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint["metrics"]["best_loss"])

        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_model_state = {
                key: value.detach().clone()
                for key, value in best_checkpoint["model_state_dict"].items()
            }

        print(f"Resuming from: {resume_path}")
        print(f"Starting at epoch {start_epoch + 1} of {num_epochs}")
    else:
        run_name = make_run_name(cfg, data.name)
        checkpoint_dir = project_root / str(cfg.training.checkpoints_dir) / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(cfg, checkpoint_dir / "config_used.yaml")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_steps = 0
        pbar = tqdm(
            data.train,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
            disable=log_every <= 0,
        )

        for batch in pbar:
            x0 = unpack_batch(batch).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = score_matching_loss(model, sde, x0)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach().cpu())
            epoch_loss += loss_val
            num_steps += 1
            global_step += 1

            if log_every > 0 and global_step % log_every == 0:
                pbar.set_postfix(
                    step_loss=f"{loss_val:.4f}",
                    epoch_avg=f"{epoch_loss / num_steps:.4f}",
                )

        epoch_avg = epoch_loss / max(num_steps, 1)
        print(f"Epoch {epoch + 1}: avg_loss={epoch_avg:.6f}")

        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "metrics": {
                "epoch_avg_loss": epoch_avg,
                "best_loss": min(best_loss, epoch_avg),
            },
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")

        if epoch_avg < best_loss:
            best_loss = epoch_avg
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

    final_artifact = {
        "run_name": run_name,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "sde_config": OmegaConf.to_container(cfg.sde, resolve=True),
        "dataset_config": OmegaConf.to_container(cfg.dataset, resolve=True),
    }
    torch.save(final_artifact, artifact_dir / f"{run_name}_final.pt")

    if best_model_state is not None:
        best_artifact = {
            "run_name": run_name,
            "model_state_dict": best_model_state,
            "model_config": OmegaConf.to_container(cfg.model, resolve=True),
            "sde_config": OmegaConf.to_container(cfg.sde, resolve=True),
            "dataset_config": OmegaConf.to_container(cfg.dataset, resolve=True),
            "metrics": {"best_loss": best_loss},
        }
        torch.save(best_artifact, artifact_dir / f"{run_name}_best.pt")

    print(f"Saved checkpoints to: {checkpoint_dir}")
    print(f"Saved model artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
