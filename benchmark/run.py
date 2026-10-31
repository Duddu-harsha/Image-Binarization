#!/usr/bin/env python3
"""
benchmark/run.py — Score the binarizer against ground truth (DIBCO format).

Usage:
    python -m benchmark.run --images data/dibco2017/Dataset --gt data/dibco2017/GT

Ground-truth files are matched by name: <stem>_gt.<ext> or <stem>.<ext>.
Compares this project against plain Otsu and plain Sauvola baselines.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.metrics import evaluate
from pipeline import binarize


def _otsu(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] == 0


def _sauvola(gray: np.ndarray, window: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    g = gray.astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(g * g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.clip(sq - mean * mean, 0, None))
    return g < mean * (1 + k * (std / r - 1))


METHODS = {
    "otsu": lambda bgr: _otsu(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
    "sauvola": lambda bgr: _sauvola(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
    "this project": lambda bgr: binarize(bgr, mode="document").binary == 0,
}


def find_pairs(images: Path, gt: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image in sorted(images.iterdir()):
        if image.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        for candidate in (gt / f"{image.stem}_gt{image.suffix}", gt / image.name,
                          gt / f"{image.stem}_gt.bmp", gt / f"{image.stem}_gt.png"):
            if candidate.exists():
                pairs.append((image, candidate))
                break
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, action="append")
    parser.add_argument("--gt", type=Path, required=True, action="append")
    args = parser.parse_args(argv)
    if len(args.images) != len(args.gt):
        parser.error("give one --gt per --images")

    pairs = [p for images, gt in zip(args.images, args.gt) for p in find_pairs(images, gt)]
    if not pairs:
        print("No image/ground-truth pairs found.", file=sys.stderr)
        return 1

    scores = {name: [] for name in METHODS}
    seconds = {name: 0.0 for name in METHODS}
    for image_path, gt_path in pairs:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        truth = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) < 128
        for name, method in METHODS.items():
            start = time.perf_counter()
            pred = method(bgr)
            seconds[name] += time.perf_counter() - start
            scores[name].append(evaluate(pred, truth))

    print(f"{len(pairs)} images  (FM, PSNR: higher is better; DRD: lower is better)\n")
    print(f"| {'method':<13} | {'FM':>6} | {'PSNR':>6} | {'DRD':>6} | {'s/img':>6} |")
    print(f"|{'-' * 15}|{'-' * 8}|{'-' * 8}|{'-' * 8}|{'-' * 8}|")
    for name, rows in scores.items():
        fm = np.mean([s.fm for s in rows])
        psnr = np.mean([s.psnr for s in rows])
        drd = np.mean([s.drd for s in rows])
        print(f"| {name:<13} | {fm:6.2f} | {psnr:6.2f} | {drd:6.2f} | {seconds[name] / len(pairs):6.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
