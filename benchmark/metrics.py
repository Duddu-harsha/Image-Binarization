"""
benchmark/metrics.py — Standard DIBCO binarization metrics.

  FM    F-measure (%) of ink pixels
  PSNR  Peak signal-to-noise ratio (dB), with ink/paper as 1/0
  DRD   Distance-Reciprocal Distortion (Lu et al., 2004) — lower is better
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Scores:
    fm: float
    psnr: float
    drd: float


def _drd_weights() -> np.ndarray:
    w = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if (i, j) != (2, 2):
                w[i, j] = 1.0 / np.hypot(i - 2, j - 2)
    return w / w.sum()


_DRD_W = _drd_weights()


def evaluate(pred_ink: np.ndarray, true_ink: np.ndarray) -> Scores:
    """Score a prediction against ground truth. Both are bool arrays, True = ink."""
    if pred_ink.shape != true_ink.shape:
        raise ValueError(f"shape mismatch: {pred_ink.shape} vs {true_ink.shape}")
    p, t = pred_ink.astype(bool), true_ink.astype(bool)

    tp = np.count_nonzero(p & t)
    precision = tp / max(np.count_nonzero(p), 1)
    recall = tp / max(np.count_nonzero(t), 1)
    fm = 100 * 2 * precision * recall / (precision + recall) if tp else 0.0

    mse = np.mean(p != t)
    psnr = float("inf") if mse == 0 else 10 * np.log10(1 / mse)

    # DRD: each flipped pixel is weighted by how much it disagrees with its 5×5 GT neighbourhood
    g, pf = t.astype(np.float64), p.astype(np.float64)
    neigh = cv2.filter2D(g, -1, _DRD_W, borderType=cv2.BORDER_REPLICATE)
    flipped = p != t
    drd_sum = np.sum((pf + neigh - 2 * pf * neigh)[flipped])
    h, w = g.shape
    hb, wb = h // 8 * 8, w // 8 * 8
    blocks = g[:hb, :wb].reshape(hb // 8, 8, wb // 8, 8).sum(axis=(1, 3))
    nubn = np.count_nonzero((blocks > 0) & (blocks < 64))   # non-uniform 8×8 GT blocks
    drd = drd_sum / max(nubn, 1)

    return Scores(fm=float(fm), psnr=float(psnr), drd=float(drd))
