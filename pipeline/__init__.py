"""
pipeline — Adaptive black-and-white conversion for documents and photos.

    from pipeline import binarize
    result = binarize(cv2.imread("page.jpg"))
    cv2.imwrite("page_bw.png", result.binary)
    print(result.report.summary())

See pipeline/binarization.py for the processing flow.
"""

from .binarization import MODES, Report, Result, binarize

__all__ = ["MODES", "Report", "Result", "binarize"]
