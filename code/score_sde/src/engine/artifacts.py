from pathlib import Path

from src.data.datasets import DATASET_SPECS


def infer_dataset_name(text: str | None) -> str | None:
    if not text:
        return None

    lowered = text.lower()
    for dataset_name in DATASET_SPECS:
        if dataset_name in lowered:
            return dataset_name
    return None


def find_latest_artifact_for_dataset(artifact_dir: Path, dataset_name: str) -> Path:
    dataset_key = dataset_name.lower()
    for pattern in ("*_best.pt", "*_final.pt"):
        matches = sorted(
            artifact_dir.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in matches:
            if infer_dataset_name(path.stem) == dataset_key:
                return path

    raise FileNotFoundError(
        f"No model artifacts found for dataset '{dataset_name}' in {artifact_dir}."
    )


def resolve_artifact_path(
    artifact_dir: Path, dataset_name: str, artifact_name: str | None = None
) -> Path:
    dataset_key = dataset_name.lower()
    if artifact_name:
        artifact_path = artifact_dir / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        inferred_dataset = infer_dataset_name(artifact_path.stem)
        if inferred_dataset and inferred_dataset != dataset_key:
            raise ValueError(
                f"Artifact '{artifact_name}' looks like dataset '{inferred_dataset}', "
                f"but '{dataset_name}' was requested."
            )
        return artifact_path

    return find_latest_artifact_for_dataset(artifact_dir, dataset_key)
