# Installing AutoDS on Windows 10 / 11

Pick **A** if you were given a download, or **B** to run it from the source code.

## A. From the downloaded release

1. Unzip `AutoDS-windows-x64.zip` somewhere permanent, such as `C:\Apps\AutoDS`.
2. Open the `AutoDS` folder and double-click **AutoDS.exe**.
3. If Windows shows **"Windows protected your PC"**, click **More info**, then
   **Run anyway**. This appears because the app is not code-signed.

Right-click `AutoDS.exe`, then **Send to**, then **Desktop (create shortcut)** for an
icon on your desktop.

## B. From the source code

1. Install Python 3.12 from <https://www.python.org/downloads/windows/>. On the first
   screen of the installer, **tick "Add python.exe to PATH"**.
2. Unzip the AutoDS project. If the zip came from `make_share_zip.py`, the helper
   files are already inside. Otherwise copy `run-autods-windows.bat` next to
   `desktop.py`.
3. Double-click **run-autods-windows.bat**. A black window shows progress. The first
   run takes 5 to 10 minutes and about 2 GB; later runs start in seconds.
4. The AutoDS window opens. Close the black window to quit.

### Make a desktop icon (optional)

After the first successful run, right-click `make-desktop-shortcut-windows.ps1` and
choose **Run with PowerShell**. An **AutoDS** icon appears on your desktop that starts
the app without the black window. If PowerShell refuses, open PowerShell in the
project folder and run:
```powershell
powershell -ExecutionPolicy Bypass -File make-desktop-shortcut-windows.ps1
```

### If the window is blank or won't open

AutoDS uses Microsoft's Edge WebView2 runtime. It is built into Windows 11 and most
Windows 10 PCs. If yours lacks it, install the **Evergreen Bootstrapper** from
Microsoft's WebView2 page. More fixes are in `TROUBLESHOOTING.md`.

### Updating

Replace the project files (keep the `.venv` folder) and run the `.bat` again. Run
`run-autods-windows.bat reinstall` to force a package reinstall.

### Where your data lives

`%LOCALAPPDATA%\AutoDS`. Delete that folder to reset the app.
