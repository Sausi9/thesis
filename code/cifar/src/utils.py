from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


def resolve_path(project_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_latest_artifact(artifact_dir: Path, preference: str) -> Path:
    patterns = {
        "best": "*_best.pt",
        "final": "*_final.pt",
        "any": "*.pt",
    }
    if preference not in patterns:
        valid = ", ".join(sorted(patterns))
        raise ValueError(
            f"Unknown artifact_preference '{preference}'. Expected one of: {valid}."
        )

    matches = sorted(
        artifact_dir.glob(patterns[preference]),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No artifacts matching {patterns[preference]} in {artifact_dir}."
        )
    return matches[0]


def find_latest_sample(samples_dir: Path, sample_type: str | None = None) -> Path:
    matches = sorted(
        samples_dir.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No sample files found in {samples_dir}.")

    if sample_type is None:
        return matches[0]

    for path in matches:
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if payload.get("sample_type") == sample_type:
            return path

    raise FileNotFoundError(f"No sample_type={sample_type!r} files found in {samples_dir}.")


def load_model_state(
    payload: dict,
    weight_type: str = "raw",
    return_weight_type: bool = False,
) -> dict | tuple[dict, str]:
    if weight_type not in {"raw", "ema"}:
        raise ValueError("weight_type must be one of: raw, ema.")

    if weight_type == "ema" and "ema_state_dict" in payload:
        state = payload["ema_state_dict"]
        loaded_weight_type = "ema"
    else:
        if "model_state_dict" not in payload:
            raise KeyError("Expected artifact/checkpoint to contain 'model_state_dict'.")
        if weight_type == "ema":
            print("Warning: requested EMA weights, but payload has no ema_state_dict. Using raw weights.")
        state = payload["model_state_dict"]
        loaded_weight_type = "raw"

    if return_weight_type:
        return state, loaded_weight_type
    return state


def make_run_name(cfg: DictConfig, dataset_name: str) -> str:
    run_stem = cfg.run_name or dataset_name
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{run_stem}_{run_id}"


def save_yaml(cfg: DictConfig, path: Path) -> None:
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def timestamped_output_path(
    *,
    output_dir: Path,
    output_name: str | None,
    default_stem: str,
    extension: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{default_stem}_{timestamp}{extension}"
    else:
        output_name = str(output_name)
        if not output_name.endswith(extension):
            output_name = f"{output_name}{extension}"

    return output_dir / output_name


def unpack_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch
