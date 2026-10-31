"""
config.py — Tunable constants for the binarizer.

The document-path values were chosen by a parameter sweep on DIBCO 2017 and
H-DIBCO 2016 (see benchmark/). Change them only with a benchmark run to back it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # ── Files ─────────────────────────────────────────────
    output_dir: Path = Path(__file__).resolve().parent / "outputs"
    pdf_render_dpi: int = 300

    # ── Document path ─────────────────────────────────────
    # Background (paper) is estimated on a copy no larger than this.
    work_long_side: int = 2048
    # Closing kernel that erases ink to reveal paper = factor × stroke width.
    background_kernel_factor: float = 2.0
    # Ink blobs smaller than factor × stroke_width² are removed as specks.
    speck_area_factor: float = 0.5

    # ── Image-type routing ────────────────────────────────
    # Analysis runs on a copy no larger than this.
    analysis_long_side: int = 1024
    # Score above which an image is treated as a photo and dithered.
    photo_score_threshold: float = 1.7
    # A mostly-dark image is a light-on-dark page if, flipped, it scores below this.
    inverted_page_score: float = 1.0
    # Std-dev below which an image is treated as blank.
    blank_std: float = 3.0

    # ── Warnings ──────────────────────────────────────────
    small_text_px: int = 12      # character height below which we warn
    low_contrast: float = 0.25   # ink/paper contrast below which we warn


cfg = Settings()
