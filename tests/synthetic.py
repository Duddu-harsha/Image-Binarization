"""
tests/synthetic.py — Synthetic test images with known ground truth.
"""

from __future__ import annotations

import cv2
import numpy as np

_LINES = [
    "The quick brown fox jumps",
    "over the lazy dog 0123456789",
    "Pack my box with five dozen",
    "liquor jugs. Sphinx of black",
]


def make_document(
    h: int = 600,
    w: int = 900,
    tint_rgb: tuple[float, float, float] = (1.0, 0.92, 0.75),
    shade: tuple[float, float] = (0.55, 1.0),
    ink: float = 40.0,
    noise: float = 4.0,
    font_scale: float = 1.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a warm-tinted, unevenly lit page of text.

    Returns:
        (bgr_image, ground_truth) where ground_truth is True on ink pixels.
    """
    rng = np.random.default_rng(seed)
    alpha = np.zeros((h, w), np.uint8)
    step = int(h / (len(_LINES) + 1))
    for i, line in enumerate(_LINES):
        cv2.putText(alpha, line, (30, step * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, 255, max(1, round(2 * font_scale / 1.2)), cv2.LINE_AA)
    a = alpha.astype(np.float32) / 255.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = shade[0] + (shade[1] - shade[0]) * (0.6 * xx / w + 0.4 * yy / h)
    gray = 235.0 * light * (1 - a) + ink * a

    bgr = gray[..., None] * np.array(tint_rgb[::-1], np.float32)
    bgr += rng.normal(0, noise, bgr.shape)
    return np.clip(bgr, 0, 255).astype(np.uint8), alpha > 127


def make_photo(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """A photo-like BGR image: smooth shaded shapes plus multi-scale texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 60 + 120 * (xx / w)
    for _ in range(12):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(30, 140)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img += np.where(d < r, rng.uniform(-80, 80) * (1 - d / r), 0)
    for sigma, amp in ((2, 25), (6, 35), (15, 40)):
        n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma)
        img += amp * n / (n.std() + 1e-6)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.merge([img, (img * 0.9).astype(np.uint8), (img * 0.8).astype(np.uint8)])
