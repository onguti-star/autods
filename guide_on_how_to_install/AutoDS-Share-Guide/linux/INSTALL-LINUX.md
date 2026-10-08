# Installing AutoDS on Linux

Tested on Ubuntu and similar. Pick **A** if you were given a download, or **B** to
run it from the source code.

## A. From the downloaded release

```bash
tar xzf AutoDS-linux-x64.tar.gz
cd AutoDS-linux
./install.sh
```

Then search for **AutoDS** in your applications menu. `install.sh` copies the app
to `~/.local/opt/AutoDS` and adds the menu entry and icon. To remove it later, run
`./uninstall.sh` from the same folder.

If it opens in your browser instead of its own window, install the Qt libraries in
`TROUBLESHOOTING.md` (first section).

## B. From the source code

1. Make sure Python 3.10 to 3.13 is installed (3.12 is best):
   ```bash
   python3 --version
   sudo apt install python3 python3-venv      # if it is missing
   ```
2. Open a terminal in the AutoDS project folder (the one containing `desktop.py`).
   If the zip came from `make_share_zip.py`, the helper script is already there.
   Otherwise copy `run-autods-linux.sh` into that folder.
3. Start it:
   ```bash
   bash run-autods-linux.sh
   ```
   The first run builds a private `.venv` and installs the packages (5 to 10 minutes,
   about 2 GB). Later runs start in seconds. Nothing is installed system-wide.
4. If the script warns that the Qt window can't load, run the `sudo apt install ...`
   line it prints, then start it again.

### Add it to your applications menu

```bash
bash run-autods-linux.sh --install-shortcut
```

Search for **AutoDS** afterwards. The icon always launches the code in your project
folder, so updating the folder updates the app. To remove the entry:
`bash run-autods-linux.sh --remove-shortcut`.

### Updating

Replace the project files with the new version (keep your `.venv`) and run the script
again. It reinstalls packages automatically when the requirements changed. Use
`--reinstall` to force it.

### Where your data lives

`~/.local/share/AutoDS`. Delete that folder to reset the app.
