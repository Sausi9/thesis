from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm


URL_MAP = {
    "cifar10": "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1",
    "ema_cifar10": "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1",
}

CKPT_MAP = {
    "cifar10": "diffusion_cifar10_model/model-790000.ckpt",
    "ema_cifar10": "ema_diffusion_cifar10_model/model-790000.ckpt",
}

MD5_MAP = {
    "cifar10": "82ed3067fd1002f5cf4c339fb80c4669",
    "ema_cifar10": "1fa350b952534ae442b1d5235cce5cd3",
}


def md5_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_checkpoint_root() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / "diffusion_models_converted"
    return Path.home() / ".cache" / "diffusion_models_converted"


def download(url: str, path: Path, chunk_size: int = 1024 * 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("content-length") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as pbar:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    pbar.update(len(chunk))
    tmp_path.replace(path)


def resolve_ddim_checkpoint(
    *,
    checkpoint_name: str,
    checkpoint_path: str | Path | None,
    cache_dir: str | Path | None,
    download_enabled: bool,
    check_md5: bool,
) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"No DDIM checkpoint found at {path}.")
        return path.resolve()

    if checkpoint_name not in URL_MAP:
        valid = ", ".join(sorted(URL_MAP))
        raise ValueError(f"Unknown DDIM checkpoint_name '{checkpoint_name}'. Expected one of: {valid}.")

    root = Path(cache_dir).expanduser() if cache_dir is not None else default_checkpoint_root()
    path = root / CKPT_MAP[checkpoint_name]
    expected_md5 = MD5_MAP[checkpoint_name]
    needs_download = not path.is_file()
    if path.is_file() and check_md5 and md5_hash(path) != expected_md5:
        print(f"DDIM checkpoint checksum mismatch at {path}; re-downloading.")
        needs_download = True

    if needs_download:
        if not download_enabled:
            raise FileNotFoundError(
                f"No DDIM checkpoint found at {path}; set ddim.download=true or provide ddim.checkpoint_path."
            )
        print(f"Downloading DDIM checkpoint {checkpoint_name} to {path}")
        download(URL_MAP[checkpoint_name], path)

    if check_md5:
        actual_md5 = md5_hash(path)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"DDIM checkpoint checksum mismatch for {path}: expected {expected_md5}, got {actual_md5}."
            )
    return path.resolve()

