#!/usr/bin/env bash
# image-binarization - Linux/macOS setup.
# Usage: ./setup.sh [--recreate]
# Creates ./venv with Python 3.10+ and installs requirements.txt.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "Python 3.10+ is required (set PYTHON=/path/to/python3)"; exit 1; }

if [[ "${1:-}" == "--recreate" ]]; then rm -rf venv; fi
[[ -x venv/bin/python ]] || "$PY" -m venv venv

venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pytest -q tests
echo
echo "Setup complete. Activate with: source venv/bin/activate"
echo "  Convert:  python main.py --input path/to/image.jpg"
echo "  Web demo: python app.py"
