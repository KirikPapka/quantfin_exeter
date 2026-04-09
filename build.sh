#!/usr/bin/env bash
# Render / CI: CPU-only PyTorch (no nvidia-* wheels) — much lower RAM than default pip torch.
set -euo pipefail
pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt
