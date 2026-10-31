"""
pipeline/illumination.py — Estimate the paper and divide it out.

Shadows, yellowed paper and colour casts all live in the "paper" layer. A
morphological closing wider than the ink strokes erases the ink and leaves
that layer; dividing by it makes the paper uniformly white so a single global
threshold can separate ink from paper.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import cfg


def odd(x: float) -> int:
    n = int(round(x))
    return n if n % 2 else n + 1


def resize_long_side(gray: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longer side is at most ``long_side``. Returns (image, scale)."""
    h, w = gray.shape
    scale = min(1.0, long_side / max(h, w))
    if scale == 1.0:
        return gray, 1.0
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA), scale


def paper_layer(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Erase dark strokes narrower than ``kernel`` and smooth what is left."""
    closed = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    )
    return cv2.medianBlur(closed, min(odd(2 * kernel + 1), 255))


def estimate_background(gray: np.ndarray, stroke_width: float) -> np.ndarray:
    """Full-resolution estimate of the paper under the ink."""
    small, scale = resize_long_side(gray, cfg.work_long_side)
    kernel = odd(max(5, cfg.background_kernel_factor * stroke_width * scale + 2))
    background = paper_layer(small, kernel)
    if scale < 1.0:
        h, w = gray.shape
        background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    return background


def flatten(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Divide out the paper so it becomes white (255) everywhere."""
    ratio = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    return np.clip(ratio * 255.0, 0, 255).astype(np.uint8)
