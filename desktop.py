#!/usr/bin/env python3
"""
AutoDS desktop launcher.

Starts the AutoDS backend on 127.0.0.1 (this computer only) and shows the UI
in its own native window (pywebview). If a native window can't be created on
this machine it falls back to a Chrome/Edge app window or your default
browser. When the window is closed the app shuts down by itself -- unless a
model is still training.

  python desktop.py                  native window (fallback: browser)
  python desktop.py --browser        use a Chrome/Edge/Brave app window instead
  python desktop.py --tab            open in a normal browser tab
  python desktop.py --no-browser     just start the server
  python desktop.py --port 9000

The packaged builds (see build_desktop.py) run this same file.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

APP_NAME = "AutoDS"
DEFAULT_PORT = 8765
NO_VISITOR_TIMEOUT = 15 * 60     # give up if no window ever connects
GRACE_SECONDS = 20               # all windows closed for this long -> quit


# ---------------------------------------------------------------- paths / env
def user_data_dir() -> str:
    override = os.environ.get("AUTODS_DATA_DIR", "").strip()
    if override:
        return override
    home = os.path.expanduser("~")
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    elif system == "Darwin":
        base = os.path.join(home, "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
    return os.path.join(base, APP_NAME)


def clean_env() -> dict:
    """Environment for launching *other* programs (the user's browser).

    The PyInstaller bootloader points LD_LIBRARY_PATH / DYLD_LIBRARY_PATH at
    the bundled libraries; a system browser that inherits them can crash on
    incompatible copies. Put back what the user originally had.
    """
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        orig = env.pop(var + "_ORIG", None)
        if orig is not None:
            env[var] = orig
        elif getattr(sys, "frozen", False):
            env.pop(var, None)
    return env


# ------------------------------------------------------------------- network
def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_port(preferred: int) -> int:
    if port_is_free(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def autods_running_on(port: int):
    """None if AutoDS isn't running on this port, else how many windows are open."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/desktop/ping", timeout=1.5) as r:
            data = json.loads(r.read().decode())
            return int(data.get("windows", 0)) if data.get("desktop") else None
    except Exception:
        return None


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


# ------------------------------------------------------------- native window
def native_available() -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec("webview") is not None
    except Exception:
        return False


def _patch_qt_downloads() -> None:
    """pywebview's own Qt download handler targets the old Qt5 API and saves files
    without an extension. Replace it: use the real file name the server sent and
    ask where to save. (AUTODS_SAVE_DIR skips the dialog -- used for testing.)"""
    try:
        from webview.platforms import qt as wq
        from qtpy.QtWidgets import QFileDialog
    except Exception:
        return

    def on_download_requested(self, download):
        name = ""
        for getter in ("suggestedFileName", "downloadFileName"):
            try:
                name = getattr(download, getter)() or name
            except Exception:
                pass
        name = os.path.basename(name) or "download"
        auto_dir = os.environ.get("AUTODS_SAVE_DIR")
        if auto_dir:
            path = os.path.join(auto_dir, name)
        else:
            start = os.path.join(os.path.expanduser("~"), "Downloads")
            start = os.path.join(start if os.path.isdir(start) else os.path.expanduser("~"), name)
            path, _ = QFileDialog.getSaveFileName(self, "Save file", start)
        if not path:
            download.cancel()
            return
        download.setDownloadDirectory(os.path.dirname(path))
        download.setDownloadFileName(os.path.basename(path))
        download.accept()

    wq.BrowserView.on_download_requested = on_download_requested


def _set_linux_app_identity(icon_path: str) -> None:
    """Without this, GNOME/KDE show a generic icon (often a settings-like gear)
    for this window: they can only show the AutoDS icon in the taskbar, dock and
    alt-tab if the window can be matched to the installed .desktop entry, or -- as
    a fallback -- if the running QApplication itself already carries the icon.
    Must run BEFORE pywebview creates its own QApplication (it reuses ours:
    QApplication.instance() or QApplication(sys.argv)), so this is called first.
    """
    try:
        from qtpy.QtGui import QIcon
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        app.setApplicationName("AutoDS")
        # Matches the "autods.desktop" file that install.sh / install_launcher.sh
        # create in ~/.local/share/applications -- lets the window manager pull
        # the icon (and name) from there instead of guessing.
        app.setDesktopFileName("autods")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception as exc:
        print(f"Could not set the Linux app icon/identity: {exc}")


def run_window(url: str) -> int:
    """Child-process entry point: show `url` in a native window until it is closed."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:      # Chromium's sandbox refuses to run as root
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")
    data_dir = user_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    icon = resource_path(os.path.join("assets", "autods.png"))
    if platform.system() == "Linux":
        _set_linux_app_identity(icon)
    import webview
    webview.settings["ALLOW_DOWNLOADS"] = True            # CSV / Excel / report / model exports
    _patch_qt_downloads()
    webview.create_window("AutoDS", url, width=1440, height=900, min_size=(980, 640))
    webview.start(
        gui="qt" if platform.system() == "Linux" else None,
        private_mode=False,                                # keep saved datasets/layout between runs
        storage_path=os.path.join(data_dir, "webview"),
        icon=icon if os.path.exists(icon) else None,
    )
    return 0


def launch_native_window(url: str):
    """Start the native window as a separate process, so a GUI problem on this
    machine can never take the server down with it."""
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--window", url]
    else:
        cmd = [sys.executable, os.path.abspath(__file__), "--window", url]
    try:
        return subprocess.Popen(cmd, env=dict(os.environ))
    except Exception as exc:
        print(f"Could not start the native window: {exc}")
        return None


# -------------------------------------------------------------- opening a UI
def _browser_candidates() -> list:
    system = platform.system()
    if system == "Windows":
        roots = [os.environ.get(k) for k in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
        rels = [r"Microsoft\Edge\Application\msedge.exe",
                r"Google\Chrome\Application\chrome.exe",
                r"BraveSoftware\Brave-Browser\Application\brave.exe"]
        paths = [os.path.join(r, rel) for rel in rels for r in roots if r]
    elif system == "Darwin":
        paths = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                 "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
                 "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    else:
        names = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
                 "microsoft-edge", "microsoft-edge-stable", "brave-browser"]
        paths = [p for p in (shutil.which(n) for n in names) if p]
    return [p for p in paths if os.path.exists(p)]


def _default_browser_id() -> str:
    try:
        r = subprocess.run(["xdg-settings", "get", "default-web-browser"], capture_output=True,
                           text=True, timeout=3, env=clean_env())
        return r.stdout.strip().lower()
    except Exception:
        return ""


def _try_commands(commands: list) -> bool:
    """Run each launcher command until one seems to have worked."""
    for cmd in commands:
        try:
            proc = subprocess.Popen(cmd, env=clean_env(), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
            time.sleep(1.5)                      # give a quick failure time to show up
            code = proc.poll()
            if code is None or code == 0:
                print(f"Opened the page with: {cmd[0]}")
                return True
            print(f"{cmd[0]} exited with code {code}; trying the next option")
        except Exception as exc:
            print(f"Could not run {cmd[0]}: {exc}")
    return False


def open_in_default_browser(url: str) -> None:
    ok = False
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
            ok = True
        elif system == "Darwin":
            ok = _try_commands([["open", url]])
        else:
            commands = []
            firefox = shutil.which("firefox") or shutil.which("firefox-esr")
            default = _default_browser_id()
            # Firefox can't run as an app window, so ask it directly for a new window.
            if firefox and ("firefox" in default or not default):
                commands.append([firefox, "--new-window", url])
            for tool in (["xdg-open", url], ["gio", "open", url], ["sensible-browser", url]):
                if shutil.which(tool[0]):
                    commands.append(tool)
            ok = _try_commands(commands)
            if not ok:
                import webbrowser
                ok = webbrowser.open(url)
    except Exception as exc:
        print(f"Browser launch failed: {exc}")
    if not ok:
        print(f"Could not open a browser automatically. Open this address yourself: {url}")


def open_ui(url: str, as_tab: bool) -> None:
    """Prefer a chromeless app window (Chrome/Edge/Brave/Chromium); otherwise
    fall back to a normal tab in the default browser."""
    if not as_tab and not os.environ.get("AUTODS_TAB"):
        for exe in _browser_candidates():
            try:
                subprocess.Popen([exe, f"--app={url}", "--window-size=1440,920"],
                                 env=clean_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                continue
    open_in_default_browser(url)


# ------------------------------------------------------------------- shutdown
def _training_active() -> bool:
    try:
        from backend import main as backend_main
        return any(job["process"].is_alive() for job in backend_main.TRAIN_JOBS.values())
    except Exception:
        return False


def _stop_training_processes() -> None:
    try:
        from backend import main as backend_main
        for job in list(backend_main.TRAIN_JOBS.values()):
            try:
                job["process"].terminate()
            except Exception:
                pass
            try:
                job["manager"].shutdown()
            except Exception:
                pass
    except Exception:
        pass


def _watchdog(server, desktop_api, started: float) -> None:
    """Stop the server once every window has been closed (and nothing is training)."""
    empty_since = None
    while not server.should_exit:
        time.sleep(2)
        now = time.time()
        live, seen_any = desktop_api.snapshot()
        if not seen_any:
            if now - started > NO_VISITOR_TIMEOUT:
                print("No window connected -- shutting down.")
                server.should_exit = True
            continue
        if live > 0 or _training_active():
            empty_since = None
            continue
        empty_since = empty_since or now
        if now - empty_since > GRACE_SECONDS:
            print("Window closed -- shutting down.")
            server.should_exit = True


# ----------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="AutoDS desktop launcher")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AUTODS_PORT", DEFAULT_PORT)))
    parser.add_argument("--no-browser", action="store_true", help="start the server only")
    parser.add_argument("--browser", action="store_true",
                        help="use a Chrome/Edge/Brave app window instead of the native window")
    parser.add_argument("--tab", action="store_true", help="open a normal browser tab")
    parser.add_argument("--data-dir", help="where sessions and logs are stored")
    parser.add_argument("--window", metavar="URL", help=argparse.SUPPRESS)   # internal: the native window process
    args = parser.parse_args()

    if args.data_dir:
        os.environ["AUTODS_DATA_DIR"] = args.data_dir
    data_dir = user_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    os.environ["AUTODS_DATA_DIR"] = data_dir
    os.environ["AUTODS_DESKTOP"] = "1"

    # A windowed (no-console) build has no stdout/stderr: send them to a log file.
    def _has_terminal() -> bool:
        try:
            return sys.stdout is not None and sys.stdout.isatty()
        except Exception:
            return False
    if sys.stdout is None or sys.stderr is None or (getattr(sys, "frozen", False) and not _has_terminal()):
        log = open(os.path.join(data_dir, "autods.log"), "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = log

    if args.window:                       # we are the native-window process
        sys.exit(run_window(args.window))

    want_native = (native_available() and not (args.browser or args.tab or args.no_browser
                   or os.environ.get("AUTODS_BROWSER") or os.environ.get("AUTODS_TAB")))

    # Already running? Show it again instead of starting a second server.
    windows = autods_running_on(args.port)
    if windows is not None:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"AutoDS is already running on port {args.port}.")
        if args.no_browser:
            return
        if want_native:
            if windows == 0:
                launch_native_window(url)
            else:
                print("Its window is already open.")
        else:
            open_ui(url, args.tab)
        return

    port = pick_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    # Only this app's own pages may call the API.
    os.environ.setdefault("ALLOWED_ORIGINS", f"http://127.0.0.1:{port},http://localhost:{port}")

    import uvicorn
    from backend import desktop_api
    from backend.main import app

    # main.py ends with a catch-all static-files mount, so our routes must go first.
    if not any(getattr(r, "path", "").startswith("/api/desktop") for r in app.router.routes):
        app.router.routes[:0] = list(desktop_api.router.routes)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                            log_config=None, loop="asyncio", http="h11", ws="none")
    server = uvicorn.Server(config)
    window_proc = []                      # the native window process, if any

    def _quit() -> None:
        for proc in window_proc:
            try:
                proc.terminate()
            except Exception:
                pass
        server.should_exit = True

    desktop_api.set_shutdown_hook(_quit)
    started = time.time()
    threading.Thread(target=_watchdog, args=(server, desktop_api, started), daemon=True).start()

    def _show_ui() -> None:
        for _ in range(600):
            if server.started:
                break
            time.sleep(0.1)
        print(f"AutoDS is running at {url}")
        if args.no_browser:
            return
        if want_native:
            began = time.time()
            proc = launch_native_window(url)
            if proc is not None:
                window_proc.append(proc)
                code = proc.wait()
                desktop_api.forget_all()      # a still-open second window checks in again within 15 s
                # Closed by the user (0) or a normal quit: the watchdog handles shutdown.
                # Died straight away: this machine can't show a native window -- use the browser.
                if code in (0, None) or time.time() - began > 30 or server.should_exit:
                    return
                print(f"The native window exited with code {code}; falling back to the browser.")
            open_ui(url, args.tab)
        else:
            open_ui(url, args.tab)

    threading.Thread(target=_show_ui, daemon=True).start()
    try:
        server.run()
    finally:
        _stop_training_processes()
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for the training worker in packaged builds
    main()
