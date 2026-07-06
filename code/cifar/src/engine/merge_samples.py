from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils import resolve_path


def load_payload(path: Path, require_complete: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No sample payload found at {path}.")
    payload = torch.load(path, map_location="cpu")
    if "samples" not in payload:
        raise KeyError(f"Payload {path} does not contain 'samples'.")
    if require_complete and payload.get("complete") is not True:
        raise ValueError(f"Payload {path} is not marked complete=true.")
    samples = payload["samples"]
    if samples.ndim != 4:
        raise ValueError(f"Expected samples [N, C, H, W] in {path}, got {tuple(samples.shape)}.")
    return payload


def normalize_output_name(output_name: str) -> str:
    output_name = str(output_name)
    if not output_name.endswith(".pt"):
        output_name = f"{output_name}.pt"
    return output_name


def shared_value(payloads: list[dict], key: str):
    first = payloads[0].get(key)
    for payload in payloads[1:]:
        if payload.get(key) != first:
            raise ValueError(f"Cannot merge: metadata key {key!r} differs across inputs.")
    return first


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    inputs = list(cfg.merge.inputs)
    if not inputs:
        raise ValueError("merge.inputs must contain at least one sample payload.")
    if cfg.merge.output_name is None:
        raise ValueError("merge.output_name must be set.")

    input_paths = [resolve_path(project_root, str(path)) for path in inputs]
    payloads = [
        load_payload(path, require_complete=bool(cfg.merge.require_complete))
        for path in input_paths
    ]

    sample_shape = tuple(payloads[0]["samples"].shape[1:])
    for path, payload in zip(input_paths, payloads, strict=True):
        shape = tuple(payload["samples"].shape[1:])
        if shape != sample_shape:
            raise ValueError(
                f"Cannot merge {path}: sample shape {shape} differs from {sample_shape}."
            )

    samples = torch.cat([payload["samples"].detach().cpu() for payload in payloads], dim=0)
    if cfg.merge.max_samples is not None:
        max_samples = int(cfg.merge.max_samples)
        if max_samples <= 0:
            raise ValueError("merge.max_samples must be positive when set.")
        samples = samples[:max_samples]

    first = payloads[0]
    output_payload = {
        key: value
        for key, value in first.items()
        if key not in {"samples", "ratio_logit_mean", "ratio_logit_std"}
    }
    output_payload.update(
        {
            "samples": samples,
            "complete": True,
            "num_completed": int(samples.shape[0]),
            "num_merged_inputs": len(input_paths),
            "merged_from": [str(path) for path in input_paths],
            "merged_input_counts": [int(payload["samples"].shape[0]) for payload in payloads],
            "merge_config": OmegaConf.to_container(cfg.merge, resolve=True),
        }
    )

    for key in (
        "sample_type",
        "feature",
        "ratio_classifier_path",
        "guidance_scale",
        "guidance_ramp",
        "guidance_start",
        "num_particles",
        "num_steps",
        "artifact_path",
        "loaded_weight_type",
    ):
        if key in first:
            output_payload[key] = shared_value(payloads, key)

    output_dir = resolve_path(project_root, str(cfg.merge.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / normalize_output_name(str(cfg.merge.output_name))
    torch.save(output_payload, output_path)

    print(f"Merged {len(input_paths)} inputs into: {output_path}")
    print(f"Total samples: {samples.shape[0]}")


if __name__ == "__main__":
    main()
