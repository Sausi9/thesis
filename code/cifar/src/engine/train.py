from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from torchvision.utils import make_grid, save_image

from src.data.dataset import build_dataloaders
from src.engine.sample import euler_maruyama_sample
from src.training.ema import ExponentialMovingAverage
from src.utils import make_run_name, resolve_device, resolve_path, save_yaml, unpack_batch


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


def clone_state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def add_ema_state(payload: dict, ema: ExponentialMovingAverage | None) -> None:
    if ema is None:
        return
    payload["ema_state_dict"] = ema.state_dict()
    payload["ema_decay"] = ema.decay


class TorchRNGState:
    def __init__(self):
        self.cpu = torch.random.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.mps = None
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.mps = torch.mps.get_rng_state()

    def restore(self) -> None:
        torch.random.set_rng_state(self.cpu)
        if self.cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda)
        if self.mps is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.set_rng_state(self.mps)


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


def optional_positive_int(value) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"Expected a positive integer or null, got {value}.")
    return value


@torch.no_grad()
def evaluate_loss(
    *,
    model,
    sde,
    dataloader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x0 = unpack_batch(batch).to(device)
        loss = score_matching_loss(model, sde, x0)
        total_loss += float(loss.detach().cpu())
        total_batches += 1

    if was_training:
        model.train()

    if total_batches == 0:
        return float("nan")
    return total_loss / total_batches


def should_run_epoch_task(epoch: int, every: int) -> bool:
    if every <= 0:
        raise ValueError(f"Expected positive epoch interval, got {every}.")
    return (epoch + 1) % every == 0


def save_training_preview(
    *,
    cfg: DictConfig,
    project_root: Path,
    run_name: str,
    epoch: int,
    model,
    ema: ExponentialMovingAverage | None,
    sde,
    device: torch.device,
    sample_shape: tuple[int, int, int],
) -> Path:
    preview_cfg = cfg.training.preview
    weight_type = str(preview_cfg.weight_type)
    if weight_type not in {"raw", "ema"}:
        raise ValueError("training.preview.weight_type must be one of: raw, ema.")

    was_training = model.training
    raw_state = None
    loaded_weight_type = "raw"
    if weight_type == "ema":
        if ema is None:
            print("Warning: requested EMA training preview, but EMA is disabled. Using raw weights.")
        else:
            raw_state = clone_state_dict_to_cpu(model.state_dict())
            model.load_state_dict(ema.shadow)
            loaded_weight_type = "ema"

    rng_state = TorchRNGState()
    try:
        torch.manual_seed(int(preview_cfg.seed))
        samples = euler_maruyama_sample(
            model=model,
            sde=sde,
            num_samples=int(preview_cfg.num_samples),
            sample_shape=sample_shape,
            num_steps=int(preview_cfg.num_steps),
            device=device,
            return_mean=bool(preview_cfg.return_mean),
            progress=False,
        )
    finally:
        rng_state.restore()
        if raw_state is not None:
            model.load_state_dict(raw_state)
        if was_training:
            model.train()
        else:
            model.eval()

    preview_dir = resolve_path(project_root, str(preview_cfg.output_dir))
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / (
        f"{run_name}_epoch_{epoch + 1:04d}_{loaded_weight_type}.png"
    )
    preview = (samples.detach().cpu().clamp(-1, 1) + 1) / 2
    grid = make_grid(preview, nrow=int(preview_cfg.nrow))
    save_image(grid, preview_path)
    return preview_path


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
    ema_enabled = bool(OmegaConf.select(cfg, "training.ema.enabled", default=False))
    ema_decay = float(OmegaConf.select(cfg, "training.ema.decay", default=0.999))
    validation_enabled = bool(
        OmegaConf.select(cfg, "training.validation.enabled", default=True)
    )
    validation_every = int(OmegaConf.select(cfg, "training.validation.every", default=1))
    validation_max_batches = optional_positive_int(
        OmegaConf.select(cfg, "training.validation.max_batches", default=None)
    )
    validation_seed = int(OmegaConf.select(cfg, "training.validation.seed", default=1234))
    preview_enabled = bool(OmegaConf.select(cfg, "training.preview.enabled", default=False))
    preview_every = int(OmegaConf.select(cfg, "training.preview.every", default=5))

    best_loss = float("inf")
    best_test_loss = float("inf")
    best_model_state = None
    best_ema_state = None
    start_epoch = 0
    global_step = 0
    ema = None

    if cfg.resume.checkpoint_path is not None:
        resume_path = resolve_checkpoint_path(project_root, str(cfg.resume.checkpoint_path))
        checkpoint_dir = resume_path.parent
        run_name = checkpoint_dir.name
        checkpoint = torch.load(resume_path, map_location=device)

        if bool(cfg.resume.strict_config):
            assert_resume_config_matches(cfg, checkpoint)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema_enabled:
            ema = ExponentialMovingAverage(model, ema_decay)
            if "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"], device=device)
            else:
                print("Resume checkpoint has no ema_state_dict. Initializing EMA from raw weights.")
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint["metrics"]["best_loss"])
        best_test_loss = float(checkpoint["metrics"].get("best_test_loss", float("inf")))

        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_model_state = clone_state_dict_to_cpu(best_checkpoint["model_state_dict"])
            if ema_enabled and "ema_state_dict" in best_checkpoint:
                best_ema_state = clone_state_dict_to_cpu(best_checkpoint["ema_state_dict"])

        print(f"Resuming from: {resume_path}")
        print(f"Starting at epoch {start_epoch + 1} of {num_epochs}")
    else:
        run_name = make_run_name(cfg, data.name)
        checkpoint_dir = project_root / str(cfg.training.checkpoints_dir) / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(cfg, checkpoint_dir / "config_used.yaml")
        if ema_enabled:
            ema = ExponentialMovingAverage(model, ema_decay)

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
            if ema is not None:
                ema.update(model)

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
        metrics = {
            "epoch_avg_loss": epoch_avg,
            "best_loss": min(best_loss, epoch_avg),
        }

        if validation_enabled and should_run_epoch_task(epoch, validation_every):
            rng_state = TorchRNGState()
            try:
                torch.manual_seed(validation_seed)
                test_avg = evaluate_loss(
                    model=model,
                    sde=sde,
                    dataloader=data.test,
                    device=device,
                    max_batches=validation_max_batches,
                )
            finally:
                rng_state.restore()
            best_test_loss = min(best_test_loss, test_avg)
            metrics["test_avg_loss"] = test_avg
            metrics["best_test_loss"] = best_test_loss
            print(
                f"Epoch {epoch + 1}: train_avg_loss={epoch_avg:.6f} "
                f"test_avg_loss={test_avg:.6f}"
            )
        else:
            metrics["best_test_loss"] = best_test_loss
            print(f"Epoch {epoch + 1}: train_avg_loss={epoch_avg:.6f}")

        if preview_enabled and should_run_epoch_task(epoch, preview_every):
            preview_path = save_training_preview(
                cfg=cfg,
                project_root=project_root,
                run_name=run_name,
                epoch=epoch,
                model=model,
                ema=ema,
                sde=sde,
                device=device,
                sample_shape=tuple(data.shape),
            )
            print(f"Saved training preview to: {preview_path}")

        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "metrics": metrics,
        }
        add_ema_state(checkpoint, ema)
        torch.save(checkpoint, checkpoint_dir / "latest.pt")

        if epoch_avg < best_loss:
            best_loss = epoch_avg
            best_model_state = clone_state_dict_to_cpu(model.state_dict())
            if ema is not None:
                best_ema_state = ema.state_dict()
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

    final_artifact = {
        "run_name": run_name,
        "model_state_dict": clone_state_dict_to_cpu(model.state_dict()),
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "sde_config": OmegaConf.to_container(cfg.sde, resolve=True),
        "dataset_config": OmegaConf.to_container(cfg.dataset, resolve=True),
        "image_shape": tuple(data.shape),
    }
    add_ema_state(final_artifact, ema)
    torch.save(final_artifact, artifact_dir / f"{run_name}_final.pt")

    if best_model_state is not None:
        best_artifact = {
            "run_name": run_name,
            "model_state_dict": best_model_state,
            "model_config": OmegaConf.to_container(cfg.model, resolve=True),
            "sde_config": OmegaConf.to_container(cfg.sde, resolve=True),
            "dataset_config": OmegaConf.to_container(cfg.dataset, resolve=True),
            "image_shape": tuple(data.shape),
            "metrics": {
                "best_loss": best_loss,
                "best_test_loss": best_test_loss,
            },
        }
        if best_ema_state is not None:
            best_artifact["ema_state_dict"] = best_ema_state
            best_artifact["ema_decay"] = ema_decay
        torch.save(best_artifact, artifact_dir / f"{run_name}_best.pt")

    print(f"Saved checkpoints to: {checkpoint_dir}")
    print(f"Saved model artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
