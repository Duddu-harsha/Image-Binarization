"""
pipeline/binarization.py — Turn any image into pure black and white.

    image ─► gray ─► classify ─┬─ blank    ─► solid white (or black)
                               ├─ photo    ─► contrast stretch ─► Floyd–Steinberg dither
                               └─ document ─► flip if light-on-dark
                                              ─► measure strokes ─► estimate paper ─► divide it out
                                              ─► Otsu threshold ─► remove specks

Every decision and measurement is recorded in the returned Report.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from config import cfg
from .analysis import classify, measure_strokes
from .illumination import estimate_background, flatten

MODES = ("auto", "document", "photo")


@dataclass
class Report:
    """What the binarizer saw and decided for one image."""
    kind: str                      # "blank" | "document" | "photo"
    mode: str                      # mode requested by the caller
    photo_score: float
    inverted: bool = False         # light-on-dark page was flipped
    stroke_width_px: float = 0.0
    char_height_px: float = 0.0
    threshold: float = 0.0         # Otsu threshold on the flattened page
    contrast: float = 0.0          # (paper - ink) / paper, 0..1
    specks_removed: int = 0
    ink_fraction: float = 0.0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}

    def summary(self) -> str:
        if self.kind == "blank":
            text = "blank page"
        elif self.kind == "photo":
            text = f"photo (score {self.photo_score:.2f}), dithered"
        else:
            text = (f"document{' (inverted)' if self.inverted else ''}: "
                    f"stroke {self.stroke_width_px:.1f}px, text {self.char_height_px:.0f}px, "
                    f"contrast {self.contrast:.2f}, {self.specks_removed} specks removed")
        return f"{text}, {self.seconds:.2f}s"


@dataclass
class Result:
    binary: np.ndarray   # uint8, 0 = black, 255 = white
    report: Report


def to_gray(image: np.ndarray) -> np.ndarray:
    """Accept grayscale, BGR or BGRA (OpenCV channel order), 8- or 16-bit."""
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise TypeError(f"expected uint8 or uint16 image, got {image.dtype}")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape {image.shape}")


def binarize(image: np.ndarray, mode: str = "auto") -> Result:
    """Convert an image to pure black and white.

    Args:
        image: grayscale, BGR or BGRA array.
        mode: "auto" (detect), "document" (threshold) or "photo" (dither).

    Returns:
        Result with the binary image (0/255) and a Report explaining it.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    start = time.perf_counter()
    gray = to_gray(image)
    seen = classify(gray)
    kind = seen.kind if mode == "auto" or seen.kind == "blank" else mode
    report = Report(kind=kind, mode=mode, photo_score=seen.photo_score)

    if kind == "blank":
        binary = np.full_like(gray, 255 if gray.mean() >= 128 else 0)
    elif kind == "photo":
        binary = _dither(gray)
    else:
        report.inverted = seen.inverted
        binary = _binarize_document(255 - gray if seen.inverted else gray, report)

    report.ink_fraction = float(np.mean(binary == 0))
    report.seconds = time.perf_counter() - start
    return Result(binary=binary, report=report)


def _binarize_document(gray: np.ndarray, report: Report) -> np.ndarray:
    stroke, _ = measure_strokes(gray)
    flat = flatten(gray, estimate_background(gray, stroke))

    threshold, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = flat <= threshold

    # Re-measure on the clean, flattened page for speck size and the report.
    stroke, char_height = measure_strokes(flat)
    min_area = max(2, round(cfg.speck_area_factor * stroke * stroke))
    ink, removed = remove_specks(ink, min_area)

    paper_level = float(np.median(flat[~ink])) if (~ink).any() else 255.0
    ink_level = float(np.median(flat[ink])) if ink.any() else paper_level
    contrast = (paper_level - ink_level) / max(paper_level, 1.0)

    report.stroke_width_px = stroke
    report.char_height_px = char_height
    report.threshold = float(threshold)
    report.contrast = contrast
    report.specks_removed = removed
    if 0 < char_height < cfg.small_text_px:
        report.warnings.append(
            f"Text is small (~{char_height:.0f}px tall); a higher-resolution scan or photo "
            "will give cleaner results.")
    if ink.any() and contrast < cfg.low_contrast:
        report.warnings.append(
            f"Ink is faint (contrast {contrast:.2f}); some strokes may be lost.")

    return np.where(ink, 0, 255).astype(np.uint8)


def remove_specks(ink: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    """Drop ink components smaller than ``min_area`` pixels. Returns (ink, count removed)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels], int(np.count_nonzero(~keep[1:]))


def _dither(gray: np.ndarray) -> np.ndarray:
    """Stretch contrast (1st–99th percentile), then Floyd–Steinberg dither."""
    lo, hi = np.percentile(gray, (1, 99))
    stretched = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255, 0, 255)
    dithered = Image.fromarray(stretched.astype(np.uint8)).convert("1")
    return np.asarray(dithered, dtype=np.uint8) * 255
