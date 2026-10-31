#!/usr/bin/env python3
"""
app.py — Web demo.

Launch:  python app.py
Opens:   http://127.0.0.1:7860  (local only; pass --share-lan to expose on your network)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg
from pipeline import MODES
from pipeline.files import SUPPORTED_EXTENSIONS, process_file
from utils.image_io import load_image, pdf_pages


def _preview(path: Path):
    """First page of the input as RGB for display."""
    if path.suffix.lower() == ".pdf":
        image, _ = next(pdf_pages(path, 100))
    else:
        image, _ = load_image(path)
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert(file_path: str | None, mode: str):
    if not file_path:
        return None, None, "Upload an image or PDF first.", None
    path = Path(file_path)
    try:
        pages = process_file(path, cfg.output_dir, mode=mode)
    except Exception as exc:
        return None, None, f"**Could not process {path.name}:** {exc}", None

    lines = []
    for page in pages:
        label = path.name if page.page is None else f"{path.name}, page {page.page}"
        lines.append(f"**{label}**: {page.report.summary()}")
        lines += [f"- {w}" for w in page.report.warnings]
    first = cv2.imread(str(pages[0].output_path), cv2.IMREAD_GRAYSCALE)
    outputs = [str(p.output_path) for p in pages]
    return _preview(path), first, "\n\n".join(lines), outputs


with gr.Blocks(title="Image Binarization", analytics_enabled=False) as demo:
    gr.Markdown(
        "# Image Binarization\n"
        "Upload a photo of a page, a scan, a screenshot or a PDF and get clean black and white. "
        "Documents are thresholded after evening out the lighting; photos are dithered."
    )
    with gr.Row():
        file_in = gr.File(label="Image or PDF", type="filepath",
                          file_types=sorted(SUPPORTED_EXTENSIONS))
        mode_in = gr.Radio(choices=list(MODES), value="auto", label="Mode")
    run = gr.Button("Convert", variant="primary")
    with gr.Row():
        before = gr.Image(label="Input", type="numpy", interactive=False)
        after = gr.Image(label="Black & white", type="numpy", interactive=False)
    report = gr.Markdown()
    downloads = gr.Files(label="Download")
    run.click(convert, inputs=[file_in, mode_in], outputs=[before, after, report, downloads])


if __name__ == "__main__":
    host = "0.0.0.0" if "--share-lan" in sys.argv else "127.0.0.1"
    demo.launch(server_name=host)
