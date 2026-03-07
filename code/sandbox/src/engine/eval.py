import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torchvision.utils import make_grid, save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from src.data.datasets import get_data
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


def find_default_artifact(artifact_dir: Path) -> Path:
    best_files = sorted(artifact_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime)
    if best_files:
        return best_files[-1]
    final_files = sorted(artifact_dir.glob("*_final.pt"), key=lambda p: p.stat().st_mtime)
    if final_files:
        return final_files[-1]
    raise FileNotFoundError(f"No model artifacts found in {artifact_dir}.")


def to_uint8_three_channel(x: torch.Tensor) -> torch.Tensor:
    x = (x.clamp(-1, 1) + 1) / 2
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = (x * 255).to(torch.uint8)
    return x


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "eval.yaml")
    eval_cfg = config["eval"]

    device = resolve_device(config.get("device", "auto"))
    artifact_dir = project_root / "artifacts" / "models"

    artifact_name = config.get("artifact_name")
    if artifact_name:
        artifact_path = artifact_dir / artifact_name
    else:
        artifact_path = find_default_artifact(artifact_dir)

    checkpoint = torch.load(artifact_path, map_location=device)
    model_cfg = checkpoint["model_config"]
    diffusion_cfg = checkpoint["diffusion_config"]

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

    T = int(diffusion_cfg["T"])
    beta_start = float(diffusion_cfg["beta_start"])
    beta_end = float(diffusion_cfg["beta_end"])
    betas = torch.linspace(beta_start, beta_end, T)
    diffusion = Diffusion(betas=betas, T=T)

    _, test_loader = get_data()
    num_test_set = len(test_loader.dataset)

    batch_size = int(eval_cfg.get("batch_size", 100))
    compute_fid = bool(eval_cfg.get("compute_fid", True))
    compute_kid = bool(eval_cfg.get("compute_kid", True))
    compute_inception_score = bool(eval_cfg.get("compute_inception_score", True))
    save_preview = bool(eval_cfg.get("save_preview", True))
    preview_num_samples = int(eval_cfg.get("preview_num_samples", 64))
    preview_nrow = int(eval_cfg.get("preview_nrow", 8))
    use_timestamp = bool(eval_cfg.get("use_timestamp", True))

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device) if compute_fid else None
    kid = KernelInceptionDistance(subset_size=1000, normalize=False).to(device) if compute_kid else None
    inception_score = InceptionScore(normalize=False).to(device) if compute_inception_score else None

    for x_real, _ in test_loader:
        x_real = to_uint8_three_channel(x_real.to(device))
        if fid is not None:
            fid.update(x_real, real=True)
        if kid is not None:
            kid.update(x_real, real=True)

    preview_samples = None
    generated = 0
    with torch.no_grad():
        while generated < num_test_set:
            current_batch = min(batch_size, num_test_set - generated)
            x_fake = diffusion.sample(model, num_samples=current_batch, device=device)
            if preview_samples is None:
                preview_samples = x_fake[:preview_num_samples].detach().cpu()
            x_fake = to_uint8_three_channel(x_fake)

            if fid is not None:
                fid.update(x_fake, real=False)
            if kid is not None:
                kid.update(x_fake, real=False)
            if inception_score is not None:
                inception_score.update(x_fake)

            generated += current_batch

    results = {
        "artifact_name": artifact_path.name,
        "num_real_samples": int(num_test_set),
        "num_fake_samples": int(num_test_set),
    }

    if fid is not None:
        results["fid"] = float(fid.compute().item())
    if kid is not None:
        kid_mean, kid_std = kid.compute()
        results["kid_mean"] = float(kid_mean.item())
        results["kid_std"] = float(kid_std.item())
    if inception_score is not None:
        is_mean, is_std = inception_score.compute()
        results["inception_score_mean"] = float(is_mean.item())
        results["inception_score_std"] = float(is_std.item())

    eval_root = project_root / "runs" / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_name = artifact_path.stem
    if use_timestamp:
        eval_name = f"{eval_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_dir = eval_root / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if save_preview and preview_samples is not None:
        preview = (preview_samples.clamp(-1, 1) + 1) / 2
        grid = make_grid(preview, nrow=preview_nrow)
        save_image(grid, eval_dir / "samples_preview.png")

    print(f"Evaluated artifact: {artifact_path.name}")
    print(f"Saved metrics to: {metrics_path}")
    for key, value in results.items():
        if key.endswith("_name"):
            continue
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
