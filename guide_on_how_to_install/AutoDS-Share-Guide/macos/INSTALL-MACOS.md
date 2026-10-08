# Installing AutoDS on macOS

Pick **A** if you were given a download, or **B** to run it from the source code.

## A. From the downloaded release

1. Unzip `AutoDS-macos-arm64.zip` (Apple Silicon: M1, M2, M3, M4).
2. Drag **AutoDS.app** into your **Applications** folder.
3. The first time, **right-click AutoDS.app and choose Open**, then click **Open** in
   the dialog. Double-clicking works normally after that.

If macOS says the app "is damaged" or "can't be opened", remove the download flag:
```bash
xattr -dr com.apple.quarantine /Applications/AutoDS.app
```

Intel Macs: there is no ready-made download. Use option B.

## B. From the source code

1. Install Python 3.12 from <https://www.python.org/downloads/macos/> (the default
   Python that comes with macOS is too old).
2. Unzip the AutoDS project. If the zip came from `make_share_zip.py`, the helper is
   already inside. Otherwise copy `run-autods-mac.command` next to `desktop.py`.
3. Allow the helper to run (first time only). In Terminal:
   ```bash
   cd path/to/autods
   chmod +x run-autods-mac.command
   ```
4. Double-click `run-autods-mac.command` (or right-click and choose Open the first
   time). A Terminal window shows progress. The first run takes 5 to 10 minutes and
   about 2 GB; later runs start in seconds.
5. The AutoDS window opens. Close the Terminal window to quit.

### Updating

Replace the project files (keep the `.venv` folder) and run the helper again. Use
`./run-autods-mac.command --reinstall` to force a package reinstall.

### Where your data lives

`~/Library/Application Support/AutoDS`. Delete it to reset the app.
