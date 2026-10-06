"""Pretrained model weights: where they come from, their licenses, and a local cache.

Base weights live in the app data directory (``settings.data_dir() / "models"``), not in the
output folder, so every output folder shares them. Downloads are verified against a pinned
SHA-256 and written atomically.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from swingvision.settings import data_dir
from swingvision.storage.fsutil import atomic_write

ULTRALYTICS_RELEASES = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0",
)
HF = "https://huggingface.co"
VITPOSE_REV = "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"
MOTIONBERT_REV = "370a9196aa3c89198b134c82476143b01c0fb32c"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    urls: tuple[str, ...]
    sha256: str | None
    license: str
    task: str
    description: str
    size_mb: float

    @property
    def path(self) -> Path:
        return weights_dir() / self.filename

    def available(self) -> bool:
        return self.path.exists()


def _yolo(name: str, sha256: str, size_mb: float, description: str) -> ModelSpec:
    filename = f"{name}.pt"
    return ModelSpec(
        name=name,
        filename=filename,
        urls=tuple(f"{base}/{filename}" for base in ULTRALYTICS_RELEASES),
        sha256=sha256,
        license="AGPL-3.0 (Ultralytics)",
        task="person detection (COCO)",
        description=description,
        size_mb=size_mb,
    )


REGISTRY: dict[str, ModelSpec] = {
    m.name: m
    for m in (
        _yolo(
            "yolo11s",
            "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
            19.3,
            "Fastest; misses more far-court players",
        ),
        _yolo(
            "yolo11m",
            "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
            40.7,
            "Default: reliable on the far player at 1920 px",
        ),
        _yolo(
            "yolo11l",
            "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f",
            51.4,
            "Slower; slightly better on small, distant players",
        ),
        ModelSpec(
            name="vitpose-base-simple",
            filename="vitpose-base-simple.safetensors",
            urls=(
                f"{HF}/usyd-community/vitpose-base-simple/resolve/{VITPOSE_REV}/model.safetensors",
            ),
            sha256="85375373893ddd3641f3912821073e53f5435f9e966e1dca59d004454bfe4fdf",
            license="Apache-2.0 (ViTPose, usyd-community on Hugging Face)",
            task="pose2d (COCO-17, top-down)",
            description="2D pose on player crops (ViTPose-B, simple decoder, 256×192)",
            size_mb=343.7,
        ),
        ModelSpec(
            name="motionbert-lite",
            filename="motionbert-lite-h36m.bin",
            urls=(
                f"{HF}/walterzhu/MotionBERT/resolve/{MOTIONBERT_REV}/checkpoint/pose3d/"
                "FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
            ),
            sha256="9811155371db4ca5d20f31a36a232d41012e12e1333882888a564d741861148f",
            license="Apache-2.0 (MotionBERT, Zhu et al.)",
            task="pose3d lifting (H36M-17)",
            description="2D keypoint sequences to 3D (MotionBERT-Lite, in-the-wild checkpoint)",
            size_mb=64.1,
        ),
    )
}


def weights_dir() -> Path:
    return data_dir() / "models"


def get(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}. Known: {', '.join(REGISTRY)}") from None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure(name: str, progress: Callable[[float], None] | None = None) -> Path:
    """Path to the model's weights, downloading (and verifying) them on first use."""
    spec = get(name)
    if spec.path.exists():
        return spec.path
    errors = []
    for url in spec.urls:
        try:
            atomic_write(spec.path, lambda tmp, url=url: _download(url, tmp, spec, progress))
            return spec.path
        except (OSError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Could not download {spec.name}:\n" + "\n".join(errors))


def _download(url: str, tmp: Path, spec: ModelSpec, progress) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "swingvision"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        h = hashlib.sha256()
        while block := resp.read(1 << 20):
            out.write(block)
            h.update(block)
            done += len(block)
            if progress and total:
                progress(done / total)
    if spec.sha256 and h.hexdigest() != spec.sha256:
        raise ValueError(f"checksum mismatch (got {h.hexdigest()[:12]}…)")
