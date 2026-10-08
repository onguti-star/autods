# Troubleshooting

Find your symptom below.

## The app opens in my browser instead of its own window

AutoDS tries a native window first. If that fails it quietly falls back to your
browser so it still works. Run it from a terminal (or the `.bat` / `.command` /
`.sh` helper) and read the lines it prints. A line such as
`The native window exited with code N; falling back to the browser` means the
window crashed, and the lines above it say why.

**Linux** (the most common cause). The Qt window needs system libraries, and a
fresh Ubuntu or an upgrade often lacks them.
```bash
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
  libegl1 libnss3 libxcomposite1 libxdamage1 libxrandr2 libxtst6
sudo apt install libasound2        # on Ubuntu 24.04 or newer use: libasound2t64
```
Fedora: `sudo dnf install xcb-util-cursor`. Arch: `sudo pacman -S xcb-util-cursor`.

If you see `You must have either QT or GTK with Python extensions installed`,
the Python you launch with does not have the window packages. Check:
```bash
.venv/bin/python3 -c "import webview, qtpy; from PyQt6 import QtWebEngineWidgets; print('ok')"
```
If that fails, run the Linux helper with `--reinstall`.

**Windows.** The window uses Microsoft's Edge WebView2 runtime. It is built into
Windows 11 and most Windows 10 machines. If yours lacks it, install the
"Evergreen Bootstrapper" from Microsoft's WebView2 page, then start AutoDS again.

**macOS.** Nothing extra is needed. If it still opens a browser, update macOS
and run with Python 3.12 from python.org.

## "AutoDS is already running on port 8765"

Another AutoDS is still alive, possibly hidden or half-closed. Find and stop it.

Linux / macOS:
```bash
lsof -i :8765          # note the PID in the second column
kill <PID>             # use kill -9 <PID> only if it refuses to quit
```
Windows (PowerShell):
```powershell
netstat -ano | findstr 8765      # the last number is the PID
taskkill /PID <PID> /F
```
Then start AutoDS again.

## The dock or taskbar shows a generic icon named "python3" (Linux)

GNOME could not match the window to the AutoDS launcher. Open
`~/.local/share/applications/autods.desktop` and make sure it contains
`StartupWMClass=python3`, then run
`update-desktop-database ~/.local/share/applications`, quit AutoDS completely,
and open it again. If it is still generic, log out and back in once. GNOME caches
these matches.

## Windows says "Windows protected your PC" (SmartScreen)

The app is not code-signed. Click **More info**, then **Run anyway**.

## macOS says the app "can't be opened" or "is damaged"

The app is not notarised. Right-click `AutoDS.app`, choose **Open**, then
**Open** again in the dialog. If it still refuses:
```bash
xattr -dr com.apple.quarantine /Applications/AutoDS.app
```
For the `.command` helper, run `chmod +x run-autods-mac.command` once first.

## `pip install` fails while installing packages

Almost always the Python version. Use **3.12**. Python 3.14 and other brand-new
releases often lack ready-made packages for CatBoost, XGBoost or LightGBM.
Delete the `.venv` folder and run the helper again with the right Python.

## Messages that look scary but are harmless

These can appear in the terminal and do not mean anything is broken:
- `Ignoring XDG_SESSION_TYPE=wayland on Gnome`
- `QXcbIntegration: Cannot create platform OpenGL context`
- `Pandas requires version ... of 'numexpr'`
- `Release of profile requested but WebEnginePage still not deleted` (on closing)

## Where is my data and how do I reset it?

| System | Folder |
|---|---|
| Linux | `~/.local/share/AutoDS` |
| macOS | `~/Library/Application Support/AutoDS` |
| Windows | `%LOCALAPPDATA%\AutoDS` |

`sessions/` holds your uploaded datasets and `autods.log` is the log. Old
sessions are removed after 7 days. To reset everything, close AutoDS and delete
the folder.

## I can't select or copy text in the window

Update to the latest version. Older builds disabled text selection. The fix is
`text_select=True` in `desktop.py`.

## The app is using a lot of disk space

Run `du -sh` on the folders in the table above. In a project folder, the usual
culprits are `.venv` (about 2 GB), `dist/` (several GB per build) and `.git` if
big data files were ever committed. See `FOR-THE-SHARER.md`.
