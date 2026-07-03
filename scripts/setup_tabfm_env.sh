#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir "git+https://github.com/google-research/tabfm.git#egg=tabfm"
python -m pip install --no-cache-dir jax[cpu] torch --extra-index-url https://download.pytorch.org/whl/cpu

echo "Environment ready. Run:"
echo "source $VENV_DIR/bin/activate"
