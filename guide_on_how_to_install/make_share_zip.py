#!/usr/bin/env python3
"""Build a clean, small zip of the AutoDS project that is safe to share.

It leaves out everything that is huge, private or rebuilt on the other person's
machine (.venv, .git, .sessions, dist, caches, datasets, models) but keeps
build/AutoDS.spec, which the desktop build needs. It also drops the run scripts
and this guide into the zip so your peers don't have to copy anything.

Usage (from anywhere):
    python make_share_zip.py /path/to/autods
    python make_share_zip.py /path/to/autods -o ~/Desktop/AutoDS-share.zip
"""
import argparse
import datetime
import fnmatch
import os
import sys
import zipfile

SKIP_DIRS = {
    ".venv", "venv", "env", ".git", ".sessions", "dist", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "catboost_info", "node_modules", ".continue",
    ".ipynb_checkpoints", ".idea", ".vscode", "stage",
}
SKIP_FILES = [
    "*.pyc", "*.pkl", "*.sqlite", "*.sqlite3", "*.db", "*.log", ".env", ".env.*", "=*",
    "*.zip", "*.tar.gz", "*.tgz", ".DS_Store", "Thumbs.db",
]
KEEP_IN_BUILD = {"AutoDS.spec"}          # the rest of build/ is generated output
MAX_FILE_MB = 25                         # anything bigger is almost certainly data

HELPERS = [                              # (path inside guide, name in project root)
    ("linux/run-autods-linux.sh", "run-autods-linux.sh"),
    ("macos/run-autods-mac.command", "run-autods-mac.command"),
    ("windows/run-autods-windows.bat", "run-autods-windows.bat"),
    ("windows/make-desktop-shortcut-windows.ps1", "make-desktop-shortcut-windows.ps1"),
]
GUIDE_DOCS = [
    "START-HERE.md", "TROUBLESHOOTING.md", "FOR-THE-SHARER.md",
    "windows/INSTALL-WINDOWS.md", "macos/INSTALL-MACOS.md", "linux/INSTALL-LINUX.md",
]


def skip_file(name):
    return any(fnmatch.fnmatch(name, pat) for pat in SKIP_FILES)


def collect(project):
    files, skipped_big = [], []
    for root, dirs, names in os.walk(project):
        rel_root = os.path.relpath(root, project)
        top = rel_root.split(os.sep)[0] if rel_root != "." else ""
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for n in sorted(names):
            rel = os.path.normpath(os.path.join(rel_root, n))
            if top == "build" and n not in KEEP_IN_BUILD:
                continue
            if skip_file(n):
                continue
            full = os.path.join(root, n)
            if os.path.getsize(full) > MAX_FILE_MB * 1024 * 1024:
                skipped_big.append(rel)
                continue
            files.append((full, rel))
    return files, skipped_big


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", help="path to the AutoDS project folder (contains desktop.py)")
    ap.add_argument("-o", "--output", help="zip file to create")
    args = ap.parse_args()

    project = os.path.abspath(os.path.expanduser(args.project))
    if not os.path.isfile(os.path.join(project, "desktop.py")):
        sys.exit(f"{project} doesn't look like the AutoDS project (no desktop.py).")
    guide = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stamp = datetime.date.today().isoformat()
    out = os.path.abspath(os.path.expanduser(args.output or f"AutoDS-share-{stamp}.zip"))
    top = "autods"

    files, skipped_big = collect(project)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for full, rel in files:
            z.write(full, f"{top}/{rel}".replace(os.sep, "/"))
        for src, dest in HELPERS:
            path = os.path.join(guide, src)
            if os.path.isfile(path):
                info = zipfile.ZipInfo(f"{top}/{dest}", date_time=datetime.datetime.now().timetuple()[:6])
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o755 if dest.endswith((".sh", ".command")) else 0o644) << 16
                with open(path, "rb") as fh:
                    z.writestr(info, fh.read())
        for doc in GUIDE_DOCS:
            path = os.path.join(guide, doc)
            if os.path.isfile(path):
                z.write(path, f"{top}/SHARE-GUIDE/{doc}")

    mb = os.path.getsize(out) / 1024 / 1024
    print(f"Created {out}  ({mb:.1f} MB, {len(files)} project files)")
    if skipped_big:
        print("Left out files over %d MB (probably data):" % MAX_FILE_MB)
        for r in skipped_big:
            print("   ", r)
    if not any(r == os.path.join("build", "AutoDS.spec") for _, r in files):
        print("Note: build/AutoDS.spec was not found, so the desktop build won't work from this zip.")


if __name__ == "__main__":
    main()
