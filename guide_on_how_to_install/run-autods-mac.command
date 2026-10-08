#!/bin/bash
# Run AutoDS from source in its own window.
# Put this file inside the AutoDS project folder (next to desktop.py) and double-click it.
# First time only: right-click it and choose Open, or run:  chmod +x run-autods-mac.command
#   ./run-autods-mac.command --reinstall   reinstall the Python packages
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
VENV="$HERE/.venv"

[ -f desktop.py ] || { echo "desktop.py not found. Put this file inside the AutoDS project folder, next to desktop.py."; read -r -p "Press Enter to close"; exit 1; }

pick_python() {
  for c in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

if ! "$VENV/bin/python3" -c "import sys" >/dev/null 2>&1; then
  PY="$(pick_python)" || {
    echo "AutoDS needs Python 3.10 to 3.13 (3.12 is best)."
    echo "Install it from https://www.python.org/downloads/macos/ and run this again."
    read -r -p "Press Enter to close"; exit 1; }
  echo "Creating environment with $($PY --version) ..."
  rm -rf "$VENV"
  "$PY" -m venv "$VENV"
fi
VPY="$VENV/bin/python3"

REQ_DESKTOP="$(mktemp)"
trap 'rm -f "$REQ_DESKTOP"' EXIT
grep -viE '^[[:space:]]*(pyinstaller|pillow)' requirements-desktop.txt > "$REQ_DESKTOP" 2>/dev/null || true
STAMP="$VENV/.autods_deps"
WANT="$(cat backend/requirements.txt "$REQ_DESKTOP" | cksum)"
[ "$1" = "--reinstall" ] && rm -f "$STAMP"
if [ "$(cat "$STAMP" 2>/dev/null)" != "$WANT" ]; then
  echo "Installing packages (first run takes several minutes) ..."
  "$VPY" -m pip install --upgrade pip >/dev/null
  "$VPY" -m pip install -r backend/requirements.txt -r "$REQ_DESKTOP"
  echo "$WANT" > "$STAMP"
fi

if ! "$VPY" -c "import webview" >/dev/null 2>&1; then
  echo "WARNING: the window library (pywebview) is missing, so AutoDS will open in your browser."
  echo "Run this again with --reinstall. See TROUBLESHOOTING.md."
fi

echo "Starting AutoDS ... (close this Terminal window to quit)"
exec "$VPY" desktop.py
