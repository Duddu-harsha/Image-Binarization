"""
utils/image_io.py — Load images and PDF pages; save 1-bit PNGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image, ImageOps

Dpi = Optional[tuple[float, float]]


def load_image(path: Path) -> tuple[np.ndarray, Dpi]:
    """Load an image as BGR uint8 (or grayscale for 16-bit/float files) plus its DPI.

    Applies EXIF rotation and composites transparency onto white.
    """
    with Image.open(path) as im:
        dpi = im.info.get("dpi")
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.asarray(im, dtype=np.float64)
            top = 65535.0 if im.mode.startswith("I;16") else max(float(arr.max()), 1.0)
            return np.clip(arr / top * 255, 0, 255).astype(np.uint8), dpi
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
        rgb = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1]), dpi


def pdf_pages(path: Path, dpi: int) -> Iterator[tuple[np.ndarray, Dpi]]:
    """Render each PDF page to BGR at ``dpi``, one page at a time."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            rgb = np.asarray(page.render(scale=dpi / 72).to_pil().convert("RGB"))
            page.close()
            yield np.ascontiguousarray(rgb[:, :, ::-1]), (dpi, dpi)
    finally:
        pdf.close()


def save_binary(binary: np.ndarray, path: Path, dpi: Dpi = None) -> Path:
    """Save a 0/255 image as a 1-bit PNG, keeping the source DPI if known."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(binary > 127)
    if dpi:
        image.save(path, dpi=(round(dpi[0]), round(dpi[1])))
    else:
        image.save(path)
    return path
