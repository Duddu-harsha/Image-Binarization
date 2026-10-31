"""
tests/test_pipeline.py — Tests for the binarizer, file I/O, CLI and benchmark metrics.

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.metrics import evaluate
from pipeline import binarize
from tests.synthetic import make_document, make_photo


def _fm(binary: np.ndarray, truth: np.ndarray) -> float:
    return evaluate(binary == 0, truth).fm


class TestDocuments:
    def test_tinted_unevenly_lit_page(self):
        """Warm tint + shading + noise, as in a phone photo of a page under indoor light."""
        img, truth = make_document()
        result = binarize(img)
        assert result.report.kind == "document"
        assert _fm(result.binary, truth) >= 90

    def test_output_is_strictly_binary(self):
        img, _ = make_document()
        out = binarize(img).binary
        assert out.dtype == np.uint8
        assert set(np.unique(out)) <= {0, 255}
        assert out.shape == img.shape[:2]

    def test_dark_mode_page_comes_out_black_on_white(self):
        img, truth = make_document(tint_rgb=(1.0, 1.0, 1.0), shade=(1.0, 1.0))
        result = binarize(255 - img)
        assert result.report.inverted
        assert _fm(result.binary, truth) >= 90

    def test_grayscale_input(self):
        img, truth = make_document()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert _fm(binarize(gray).binary, truth) >= 90

    def test_isolated_specks_removed(self):
        img, truth = make_document(noise=0)
        rng = np.random.default_rng(1)
        ys, xs = rng.integers(5, img.shape[0] - 5, 200), rng.integers(5, img.shape[1] - 5, 200)
        far = ~cv2.dilate(truth.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)[ys, xs]
        img[ys[far], xs[far]] = 0
        out = binarize(img).binary
        assert np.all(out[ys[far], xs[far]] == 255)
        assert _fm(out, truth) >= 90

    def test_report_measures_strokes(self):
        img, _ = make_document()
        r = binarize(img).report
        assert 1.5 <= r.stroke_width_px <= 6
        assert 15 <= r.char_height_px <= 45
        assert 0 < r.ink_fraction < 0.3

    def test_small_text_warns(self):
        img, _ = make_document(h=150, w=300, font_scale=0.3)
        assert any("small" in w for w in binarize(img).report.warnings)


class TestRouting:
    def test_photo_is_dithered(self):
        img = make_photo()
        result = binarize(img)
        assert result.report.kind == "photo"
        assert set(np.unique(result.binary)) <= {0, 255}
        # dithering keeps overall tone; thresholding into outlines would not
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = np.percentile(gray, [1, 99])
        expected = np.clip((gray - lo) / (hi - lo), 0, 1).mean()
        assert abs(result.binary.mean() / 255 - expected) < 0.1

    def test_mode_override(self):
        assert binarize(make_photo(), mode="document").report.kind == "document"
        img, _ = make_document()
        assert binarize(img, mode="photo").report.kind == "photo"

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            binarize(make_photo(), mode="banana")

    @pytest.mark.parametrize("value, expected", [(255, 255), (250, 255), (0, 0), (5, 0)])
    def test_blank_pages(self, value, expected):
        img = np.full((200, 300, 3), value, np.uint8)
        result = binarize(img)
        assert result.report.kind == "blank"
        assert np.all(result.binary == expected)


class TestFiles:
    def test_png_roundtrip_keeps_dpi_and_is_1bit(self, tmp_path):
        from pipeline.files import process_file
        img, _ = make_document()
        src = tmp_path / "page.png"
        Image.fromarray(img[:, :, ::-1]).save(src, dpi=(300, 300))
        [page] = process_file(src, tmp_path / "out")
        with Image.open(page.output_path) as out:
            assert out.mode == "1"
            assert round(out.info["dpi"][0]) == 300
        assert page.output_path.name == "page_bw.png"

    def test_pdf_pages(self, tmp_path):
        from pipeline.files import process_file
        img, _ = make_document()
        pdf = tmp_path / "doc.pdf"
        pil = Image.fromarray(img[:, :, ::-1])
        pil.save(pdf, save_all=True, append_images=[pil], resolution=150)
        pages = process_file(pdf, tmp_path / "out")
        assert [p.output_path.name for p in pages] == ["doc_p1_bw.png", "doc_p2_bw.png"]
        assert all(p.output_path.exists() for p in pages)

    def test_unsupported_file(self, tmp_path):
        from pipeline.files import process_file
        bad = tmp_path / "notes.txt"
        bad.write_text("hello")
        with pytest.raises(ValueError):
            process_file(bad, tmp_path / "out")

    def test_cli(self, tmp_path):
        import main
        img, _ = make_document()
        cv2.imwrite(str(tmp_path / "a.png"), img)
        cv2.imwrite(str(tmp_path / "b.jpg"), make_photo())
        code = main.main(["--input", str(tmp_path), "--output-dir", str(tmp_path / "out")])
        assert code == 0
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a_bw.png", "b_bw.png"]


class TestMetrics:
    def test_perfect(self):
        truth = np.zeros((32, 32), bool)
        truth[10:20, 5:25] = True
        s = evaluate(truth, truth)
        assert s.fm == 100 and s.drd == 0 and s.psnr == float("inf")

    def test_all_wrong(self):
        truth = np.zeros((32, 32), bool)
        truth[10:20, 5:25] = True
        s = evaluate(~truth, truth)
        assert s.fm == 0 and s.drd > 0

    def test_partial(self):
        truth = np.zeros((32, 32), bool)
        truth[10:20, 0:20] = True
        pred = np.zeros_like(truth)
        pred[10:20, 0:10] = True
        s = evaluate(pred, truth)
        assert s.fm == pytest.approx(100 * 2 * 1.0 * 0.5 / 1.5)
