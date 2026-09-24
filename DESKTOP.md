# AutoDS desktop app

AutoDS runs as a normal program on your computer: it starts a private server
(reachable only from your machine) and shows the interface in **its own native
window** -- no browser, address bar or tabs involved. Nothing is uploaded
anywhere, and it works with no internet connection. Closing the window shuts it
down (unless a model is still training).

If a native window can't be created on some machine, it automatically falls back
to a Chrome/Edge app window or the default browser, so it still opens.

## Using a download

**Linux** – `tar xzf AutoDS-linux-x64.tar.gz`, then `cd AutoDS-linux && ./install.sh`
and open **AutoDS** from your applications menu. (`./uninstall.sh` removes it.)

**Windows** – unzip `AutoDS-windows-x64.zip` and double-click `AutoDS\AutoDS.exe`.
Windows may show a "SmartScreen" warning because the app isn't code-signed:
click *More info → Run anyway*.

**macOS** – unzip `AutoDS-macos-arm64.zip`, drag `AutoDS.app` to Applications.
First launch: right-click → *Open* (the app isn't notarised).

On Windows the window uses the Edge WebView2 runtime that ships with Windows 10/11;
on macOS the built-in WebKit; on Linux a Qt WebEngine that is bundled inside the
download (that is most of its size). The app needs no browser installed.

Your data (uploaded sessions and a log file) lives in:
`~/.local/share/AutoDS` (Linux), `%LOCALAPPDATA%\AutoDS` (Windows),
`~/Library/Application Support/AutoDS` (macOS).

## Running from source (no build)

    pip install -r backend/requirements.txt -r requirements-desktop.txt
    sudo apt install libxcb-cursor0        # Linux only, needed by the Qt window
    python desktop.py            # native window
    python desktop.py --browser  # Chrome/Edge app window instead
    python desktop.py --tab      # normal browser tab
    python backend/run.py        # the old way (dev server on port 8000)

## Building the installers yourself

PyInstaller can't cross-compile, so build on the OS you want to target:

    sudo apt install libxcb-cursor0   # Linux only; gets bundled into the app
    pip install -r backend/requirements.txt -r requirements-desktop.txt
    python build_desktop.py           # result in dist/release/
    python build_desktop.py --lite    # without CatBoost, ~150 MB smaller

## Building all three with GitHub (free, no credit card)

1. Push the project to GitHub (a public repository gets free build minutes).
2. **Actions → Build desktop apps → Run workflow** builds Linux, Windows and
   macOS versions; download them from the run's *Artifacts*.
3. To publish a release people can download:

       git tag v1.0.0
       git push origin v1.0.0

   The workflow attaches the three files to a GitHub Release automatically.

## Options

| Flag / variable | Meaning |
|---|---|
| `--port N` / `AUTODS_PORT` | preferred port (default 8765; another is used if busy) |
| `--browser` / `AUTODS_BROWSER=1` | use a Chrome/Edge app window instead of the native one |
| `--tab` / `AUTODS_TAB=1` | use a normal browser tab |
| `--no-browser` | start the server only |
| `--data-dir PATH` / `AUTODS_DATA_DIR` | where sessions and the log are kept |

## Sharing the download (GitHub size limits)

GitHub refuses files over **100 MB inside the repository**, so never commit
`dist/` or the `.tar.gz`/`.zip` files (they are in `.gitignore`). Attach them to a
**Release** instead: Releases accept files up to **2 GB each**. The workflow above
does this for you when you push a `v*` tag, or upload by hand under
*Releases → Draft a new release → attach binaries*.
