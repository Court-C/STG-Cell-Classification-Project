#!/usr/bin/env bash
# One-command environment bootstrap for macOS/Linux: creates .venv at the
# project root if it doesn't already exist, then installs/updates
# requirements.txt into it. Safe to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating .venv ..."
    python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Done. Activate with: source .venv/bin/activate"
