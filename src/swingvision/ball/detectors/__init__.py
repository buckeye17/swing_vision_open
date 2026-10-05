"""Pluggable ball detectors. Importing this package registers the built-in ones."""

from pathlib import Path

from swingvision.ball.detectors import motion, unet, yolo  # noqa: F401
from swingvision.ball.detectors.base import (
    DETECTORS,
    BallDetector,
    DetectorSpec,
    make_ball_detector,
    register,
)

__all__ = [
    "DETECTORS",
    "BallDetector",
    "DetectorSpec",
    "make_ball_detector",
    "register",
    "resolve_spec",
]


def resolve_spec(spec: str, output_root: Path | None) -> str:
    """``auto`` → the newest trained U-Net in the output folder, else ``motion``."""
    if spec.strip() != "auto":
        return spec.strip()
    if output_root is not None:
        cards = unet.list_weights(output_root)
        cards = [c for c in cards if c.get("default", True)]
        if cards:
            newest = max(cards, key=lambda c: c.get("created_at", ""))
            return f"unet:{newest['name']}"
    return "motion"
