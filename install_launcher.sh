#!/usr/bin/env bash
# Adds an "AutoDS" icon to your applications menu that runs THIS project folder
# (no build needed). Run it once:
#     bash install_launcher.sh                     # auto-detects .venv / venv
#     bash install_launcher.sh /path/to/venv/bin/python   # or name the Python yourself
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$1" ]; then PY="$1"
elif [ -x "$HERE/.venv/bin/python" ]; then PY="$HERE/.venv/bin/python"
elif [ -x "$HERE/venv/bin/python" ]; then PY="$HERE/venv/bin/python"
elif [ -x "$HERE/../.venv/bin/python" ]; then PY="$(cd "$HERE/.." && pwd)/.venv/bin/python"
else PY="$(command -v python3)"; fi

if ! "$PY" -c "import fastapi, uvicorn, pandas, sklearn" 2>/dev/null; then
  echo "The Python at $PY is missing AutoDS's packages."
  echo "Install them with:  $PY -m pip install -r \"$HERE/backend/requirements.txt\""
  echo "or pass the Python of the virtual environment you already set up."
  exit 1
fi

BIN="$HOME/.local/bin"; APPS="$HOME/.local/share/applications"; DATA="$HOME/.local/share/AutoDS"
mkdir -p "$BIN" "$APPS" "$DATA"

# Small launcher script (keeps paths with spaces, like "DATA SCIENCE", working)
cat > "$BIN/autods" <<LAUNCH
#!/bin/sh
cd "$HERE" || exit 1
exec "$PY" desktop.py "\$@" >> "$DATA/launcher.log" 2>&1
LAUNCH
chmod +x "$BIN/autods"

cat > "$APPS/autods.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=AutoDS
Comment=Automated data science console
Exec=$BIN/autods
Icon=$HERE/assets/autods.png
Terminal=false
Categories=Education;Science;Development;
StartupWMClass=AutoDS
DESK
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true

echo "Done. Press the Super (Windows) key, type  AutoDS  and open it."
echo "Or run:  $BIN/autods"
echo "If it ever fails to open, the error is in: $DATA/launcher.log"
