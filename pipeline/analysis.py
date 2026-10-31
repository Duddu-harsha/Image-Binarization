"""
pipeline/analysis.py — Look at an image before binarizing it.

  • measure_strokes — typical ink stroke width and character height
  • classify        — blank / document / photo, and whether the page is
                      light-on-dark (e.g. a dark-mode screenshot)

Photo detection: after the paper layer is estimated, a document's paper is
smooth and its pixels are either paper or ink. A photo's "paper" still holds
structure (objects, texture) and it has many mid-tones. The photo score
combines both signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import cfg
from .illumination import odd, paper_layer, resize_long_side


@dataclass(frozen=True)
class Classification:
    kind: str            # "blank" | "document" | "photo"
    photo_score: float   # >= cfg.photo_score_threshold means photo
    inverted: bool       # True if the background is dark (used when treated as a document)


def otsu_ink(gray: np.ndarray) -> np.ndarray:
    """Rough ink mask: pixels at or below Otsu's threshold."""
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] == 0


def measure_strokes(gray: np.ndarray) -> tuple[float, float]:
    """Return (stroke_width, char_height) in pixels.

    Stroke width is twice the median distance-to-paper along the ink's ridge.
    Character height is the median height of glyph-sized ink components
    (0.0 if none are found).
    """
    ink = otsu_ink(gray).astype(np.uint8)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    ridge = (ink > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    stroke = 2.0 * float(np.median(dist[ridge])) if ridge.any() else 2.0

    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    glyphs = (areas >= max(4.0, stroke * stroke)) & (heights < gray.shape[0] / 4)
    char_height = float(np.median(heights[glyphs])) if glyphs.any() else 0.0
    return max(stroke, 1.0), char_height


def classify(gray: np.ndarray) -> Classification:
    small, _ = resize_long_side(gray, cfg.analysis_long_side)
    if float(small.std()) < cfg.blank_std:
        return Classification("blank", 0.0, False)

    # The background is the majority class. A mostly-dark image is either a
    # light-on-dark page (which looks like a clean page once flipped) or a dark photo.
    dark_majority = bool(otsu_ink(small).mean() > 0.5)
    score = _photo_score(small)
    if dark_majority:
        flipped = _photo_score(255 - small)
        if flipped < cfg.inverted_page_score:
            score = flipped
    kind = "photo" if score >= cfg.photo_score_threshold else "document"
    return Classification(kind, score, dark_majority)


def _photo_score(small: np.ndarray) -> float:
    stroke, _ = measure_strokes(small)
    kernel = odd(max(7, 0.03 * max(small.shape), 3 * stroke))
    paper = paper_layer(small, kernel).astype(np.float32)

    structure = float(np.std(cv2.GaussianBlur(paper, (0, 0), 2) - cv2.GaussianBlur(paper, (0, 0), 8)))
    spread = float(np.percentile(small, 95) - np.percentile(small, 5)) + 1.0
    ratio = small.astype(np.float32) / np.maximum(paper, 1.0)
    midtones = float(((ratio > 0.5) & (ratio <= 0.85)).mean())

    return (100.0 * structure / spread) / 3.3 + midtones / 0.3
