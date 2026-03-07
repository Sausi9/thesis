from datetime import datetime
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid, save_image

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


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "sample.yaml")

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

    num_samples = int(config["sampling"]["num_samples"])
    nrow = int(config["sampling"]["nrow"])
    output_name = str(config["sampling"].get("output_name", "latest.png"))
    use_timestamp = bool(config["sampling"].get("use_timestamp", False))
    show_samples = bool(config["sampling"].get("show", True))

    with torch.no_grad():
        samples = diffusion.sample(model, num_samples=num_samples, device=device)

    samples = (samples.clamp(-1, 1) + 1) / 2

    output_dir = project_root / "runs" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_timestamp:
        stem = Path(output_name).stem
        suffix = Path(output_name).suffix or ".png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{stem}_{timestamp}{suffix}"
    else:
        output_path = output_dir / output_name

    grid = make_grid(samples, nrow=nrow)
    save_image(grid, output_path)
    print(f"Saved samples to: {output_path}")
    print(f"Loaded artifact: {artifact_path.name}")

    if show_samples:
        Image.open(output_path).show()


if __name__ == "__main__":
    main()
