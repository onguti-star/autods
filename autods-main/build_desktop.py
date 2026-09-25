#!/usr/bin/env python3
"""
Build the AutoDS desktop app with PyInstaller for the OS you are running on.

    pip install -r backend/requirements.txt -r requirements-desktop.txt
    python build_desktop.py            # full build
    python build_desktop.py --lite     # skip CatBoost (about 150 MB smaller)

Output goes to dist/release/ (a .tar.gz on Linux, a .zip on Windows/macOS).
PyInstaller cannot cross-compile: build on Linux for Linux, on Windows for
Windows, on macOS for macOS. The GitHub Actions workflow does all three.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SYSTEM = platform.system()
ARCH = {"x86_64": "x64", "AMD64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())


def run_pyinstaller(lite: bool) -> None:
    import PyInstaller.__main__ as pyi

    sep = os.pathsep
    args = [
        os.path.join(ROOT, "desktop.py"),
        "--name", "AutoDS",
        "--noconfirm", "--clean", "--onedir",
        "--paths", ROOT,
        "--distpath", os.path.join(ROOT, "dist"),
        "--workpath", os.path.join(ROOT, "build"),
        "--specpath", os.path.join(ROOT, "build"),
        "--add-data", f"{os.path.join(ROOT, 'frontend')}{sep}frontend",
        "--add-data", f"{os.path.join(ROOT, 'backend', 'geo_data')}{sep}backend/geo_data",
        "--add-data", f"{os.path.join(ROOT, 'assets')}{sep}assets",
        # uvicorn picks its loop/protocol implementations dynamically
        "--collect-submodules", "uvicorn",
        "--hidden-import", "h11",
        # boosting libraries ship native libraries PyInstaller must be told about
        # (collect-binaries/data only: --collect-all imports every submodule,
        #  including xgboost.testing, which needs pytest + hypothesis)
        "--collect-binaries", "xgboost", "--collect-data", "xgboost",
        "--collect-binaries", "lightgbm", "--collect-data", "lightgbm",
        "--hidden-import", "python_multipart",
        "--hidden-import", "multipart",
        "--hidden-import", "openpyxl",
        "--hidden-import", "xlsxwriter",
        "--hidden-import", "pymysql",
        "--hidden-import", "psycopg2",
        "--hidden-import", "sqlalchemy.dialects.sqlite",
        "--hidden-import", "sqlalchemy.dialects.mysql",
        "--hidden-import", "sqlalchemy.dialects.postgresql",
        "--collect-submodules", "mlxtend.frequent_patterns",
        # native window (pywebview): the backend is picked by name at runtime
        "--hidden-import", "webview",
        # things AutoDS never needs at runtime
        "--exclude-module", "tkinter",
        "--exclude-module", "IPython",
        "--exclude-module", "pytest",
        "--exclude-module", "matplotlib",
    ]
    if not lite:
        args += ["--collect-binaries", "catboost", "--collect-data", "catboost"]
    else:
        args += ["--exclude-module", "catboost"]

    if SYSTEM == "Linux":
        args += ["--hidden-import", "webview.platforms.qt", "--hidden-import", "qtpy",
                 "--hidden-import", "PyQt6.QtWebEngineWidgets", "--hidden-import", "PyQt6.QtWebChannel"]
    elif SYSTEM == "Windows":
        args += ["--hidden-import", "webview.platforms.edgechromium",
                 "--hidden-import", "webview.platforms.winforms"]
    else:
        args += ["--hidden-import", "webview.platforms.cocoa"]

    icon_ico = os.path.join(ROOT, "assets", "autods.ico")
    icon_png = os.path.join(ROOT, "assets", "autods.png")
    if SYSTEM == "Windows":
        args += ["--noconsole", "--icon", icon_ico]
    elif SYSTEM == "Darwin":
        args += ["--windowed", "--icon", icon_png,
                 "--osx-bundle-identifier", "app.autods.desktop"]
    pyi.run(args)


def package() -> str:
    dist = os.path.join(ROOT, "dist")
    out = os.path.join(dist, "release")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    readme = os.path.join(ROOT, "DESKTOP.md")

    if SYSTEM == "Linux":
        stage = os.path.join(dist, "stage")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        shutil.copytree(os.path.join(dist, "AutoDS"), os.path.join(stage, "AutoDS"), symlinks=True)
        for f in ("install.sh", "uninstall.sh"):
            shutil.copy2(os.path.join(ROOT, "packaging", "linux", f), stage)
        shutil.copy2(os.path.join(ROOT, "assets", "autods.png"), stage)
        if os.path.exists(readme):
            shutil.copy2(readme, stage)
        path = os.path.join(out, f"AutoDS-linux-{ARCH}.tar.gz")
        with tarfile.open(path, "w:gz") as tar:
            tar.add(stage, arcname="AutoDS-linux")
        shutil.rmtree(stage)
    elif SYSTEM == "Windows":
        path = os.path.join(out, f"AutoDS-windows-{ARCH}.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            base = os.path.join(dist, "AutoDS")
            for folder, _, files in os.walk(base):
                for name in files:
                    full = os.path.join(folder, name)
                    z.write(full, os.path.join("AutoDS", os.path.relpath(full, base)))
    else:  # macOS: zip the .app with ditto so symlinks and permissions survive
        path = os.path.join(out, f"AutoDS-macos-{ARCH}.zip")
        subprocess.check_call(["ditto", "-c", "-k", "--keepParent",
                               os.path.join(dist, "AutoDS.app"), path])
    return path


def preflight() -> None:
    if SYSTEM != "Linux":
        return
    try:
        found = "libxcb-cursor" in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    except Exception:
        found = True
    if not found:
        sys.exit("Missing system library for the native window. Install it and run this again:\n"
                 "    sudo apt install libxcb-cursor0")
    try:
        import PyQt6.QtWebEngineWidgets  # noqa: F401
        import webview  # noqa: F401
    except Exception as exc:
        sys.exit(f"The native-window packages are missing ({exc}).\n"
                 "    pip install -r requirements-desktop.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true", help="leave out CatBoost to shrink the download")
    ap.add_argument("--no-archive", action="store_true", help="skip the .tar.gz/.zip step")
    a = ap.parse_args()
    preflight()
    run_pyinstaller(a.lite)
    if not a.no_archive:
        print("Created:", package())


if __name__ == "__main__":
    main()
