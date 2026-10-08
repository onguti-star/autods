#!/usr/bin/env bash
# Run AutoDS from source in its own window.
# Put this file inside the AutoDS project folder (next to desktop.py), then:
#   bash run-autods-linux.sh                    start AutoDS
#   bash run-autods-linux.sh --reinstall        reinstall the Python packages
#   bash run-autods-linux.sh --install-shortcut add AutoDS to the applications menu
#   bash run-autods-linux.sh --remove-shortcut  remove that menu entry
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VENV="$HERE/.venv"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"

[ -f desktop.py ] || { echo "desktop.py not found. Put this script inside the AutoDS project folder, next to desktop.py."; exit 1; }

if [ "$1" = "--remove-shortcut" ]; then
  rm -f "$HOME/.local/bin/autods" "$DATA/applications/autods.desktop"
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DATA/applications" 2>/dev/null || true
  echo "Shortcut removed."; exit 0
fi

# ---- 1. find a suitable Python (3.10 to 3.13) ----
pick_python() {
  for c in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

# ---- 2. create the private environment if it is missing or broken ----
if ! "$VENV/bin/python3" -c "import sys" >/dev/null 2>&1; then
  PY="$(pick_python)" || { echo "Need Python 3.10 to 3.13. Install it, e.g.: sudo apt install python3 python3-venv"; exit 1; }
  echo "Creating environment with $($PY --version) ..."
  rm -rf "$VENV"
  "$PY" -m venv "$VENV" || { echo "Could not create a venv. On Ubuntu/Debian run: sudo apt install python3-venv"; exit 1; }
fi
VPY="$VENV/bin/python3"

# ---- 3. install packages (only when needed) ----
REQ_DESKTOP="$(mktemp)"
trap 'rm -f "$REQ_DESKTOP"' EXIT
# requirements-desktop.txt also lists build-only tools (PyInstaller, Pillow); skip those
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

# ---- 4. check that the native window can start ----
if ! "$VPY" -c "import webview, qtpy" >/dev/null 2>&1; then
  echo
  echo "WARNING: the window packages (pywebview / Qt) are not installed in .venv, so AutoDS"
  echo "will open in your browser. Run this script again with --reinstall."
  echo
elif ! "$VPY" -c "from PyQt6 import QtWebEngineWidgets" >/dev/null 2>&1; then
  cat <<'MSG'

WARNING: the Qt window could not load, so AutoDS will open in your browser.
This is usually missing system libraries. Install them with:

  sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
    libegl1 libnss3 libxcomposite1 libxdamage1 libxrandr2 libxtst6
  sudo apt install libasound2     # Ubuntu 24.04 or newer: libasound2t64

Then start AutoDS again. See TROUBLESHOOTING.md for other distributions.

MSG
fi

# ---- 5. optional: applications-menu shortcut ----
if [ "$1" = "--install-shortcut" ]; then
  mkdir -p "$HOME/.local/bin" "$DATA/applications" "$DATA/AutoDS"
  {
    echo '#!/bin/sh'
    printf 'cd %q || exit 1\n' "$HERE"
    printf 'exec %q desktop.py "$@" >> %q 2>&1\n' "$VPY" "$DATA/AutoDS/launcher.log"
  } > "$HOME/.local/bin/autods"
  chmod +x "$HOME/.local/bin/autods"
  cat > "$DATA/applications/autods.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=AutoDS
Comment=Automated data science workbench
Exec=$HOME/.local/bin/autods
Icon=$HERE/assets/autods.png
Terminal=false
Categories=Education;Science;Development;
# GNOME matches the window to this launcher by its class name. If the dock shows a
# generic icon, see TROUBLESHOOTING.md.
StartupWMClass=python3
DESK
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DATA/applications" 2>/dev/null || true
  echo "Done. Search for 'AutoDS' in your applications menu."
  exit 0
fi

exec "$VPY" desktop.py
