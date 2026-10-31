"""
pipeline/files.py — Run the binarizer on image and PDF files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import cfg
from utils.image_io import load_image, pdf_pages, save_binary
from .binarization import Report, binarize

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


@dataclass
class PageResult:
    source: Path
    page: Optional[int]      # 1-based page number for PDFs, None for images
    output_path: Path
    report: Report


def process_file(path: Path, output_dir: Optional[Path] = None, mode: str = "auto") -> list[PageResult]:
    """Binarize an image (one result) or every page of a PDF (one result per page).

    Outputs are 1-bit PNGs named ``<stem>_bw.png`` or ``<stem>_p<N>_bw.png``.
    """
    path = Path(path)
    out_dir = Path(output_dir) if output_dir else cfg.output_dir
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.name}")

    if ext in PDF_EXTENSIONS:
        results = []
        for number, (image, dpi) in enumerate(pdf_pages(path, cfg.pdf_render_dpi), start=1):
            out = out_dir / f"{path.stem}_p{number}_bw.png"
            results.append(_process(image, dpi, path, number, out, mode))
        return results

    image, dpi = load_image(path)
    return [_process(image, dpi, path, None, out_dir / f"{path.stem}_bw.png", mode)]


def _process(image, dpi, source: Path, page: Optional[int], out: Path, mode: str) -> PageResult:
    result = binarize(image, mode=mode)
    save_binary(result.binary, out, dpi)
    return PageResult(source=source, page=page, output_path=out, report=result.report)
