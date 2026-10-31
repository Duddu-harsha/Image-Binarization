#!/usr/bin/env python3
"""
main.py — Command-line entry point.

Usage:
    python main.py --input page.jpg
    python main.py --input scans/ --output-dir clean/
    python main.py --input book.pdf --mode document --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg
from pipeline import MODES
from pipeline.files import SUPPORTED_EXTENSIONS, process_file


def gather_files(input_path: Path) -> list[Path]:
    """A single supported file, or every supported file directly inside a directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir()
                      if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert images and PDFs to clean black and white.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="image, PDF, or directory")
    parser.add_argument("--output-dir", "-o", type=Path, default=cfg.output_dir,
                        help=f"where to write results (default: {cfg.output_dir.name}/)")
    parser.add_argument("--mode", choices=MODES, default="auto",
                        help="auto-detect (default), force document thresholding, or force photo dithering")
    parser.add_argument("--json", action="store_true", help="print one JSON report per output instead of text")
    args = parser.parse_args(argv)

    files = gather_files(args.input)
    if not files:
        print(f"No supported files found at {args.input}", file=sys.stderr)
        return 1

    failures = 0
    for path in files:
        try:
            pages = process_file(path, args.output_dir, mode=args.mode)
        except Exception as exc:  # report and continue with the next file
            failures += 1
            print(f"FAILED {path.name}: {exc}", file=sys.stderr)
            continue
        for page in pages:
            if args.json:
                print(json.dumps({"input": str(page.source), "page": page.page,
                                  "output": str(page.output_path), **page.report.as_dict()}))
            else:
                label = path.name if page.page is None else f"{path.name} p{page.page}"
                print(f"{label} -> {page.output_path.name}  [{page.report.summary()}]")
                for warning in page.report.warnings:
                    print(f"    ! {warning}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
