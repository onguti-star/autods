#!/usr/bin/env bash
# Removes the app. Your data (sessions, logs) stays in ~/.local/share/AutoDS unless you delete it.
rm -rf "$HOME/.local/opt/AutoDS" \
       "$HOME/.local/share/applications/autods.desktop" \
       "$HOME/.local/share/icons/hicolor/256x256/apps/autods.png" \
       "$HOME/.local/bin/autods"
echo "AutoDS removed."
