#!/usr/bin/env bash
# Installs AutoDS for the current user (no sudo needed).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.local/opt/AutoDS"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/256x256/apps"

mkdir -p "$APPS" "$ICONS" "$HOME/.local/bin"
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -r "$HERE/AutoDS" "$DEST"
chmod +x "$DEST/AutoDS"
cp "$HERE/autods.png" "$ICONS/autods.png"
ln -sf "$DEST/AutoDS" "$HOME/.local/bin/autods"

cat > "$APPS/autods.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=AutoDS
Comment=Automated data science console
Exec=$DEST/AutoDS
Icon=autods
Terminal=false
Categories=Education;Science;Development;
StartupWMClass=AutoDS
DESK

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true
echo "AutoDS installed. Open it from your applications menu (search 'AutoDS'),"
echo "or run:  $DEST/AutoDS"
