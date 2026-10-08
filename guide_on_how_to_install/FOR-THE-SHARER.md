# For whoever is sharing AutoDS

Two ways to hand AutoDS to people. Use the first for most peers and the second for
anyone who wants the source or has an unusual machine.

## 1. Publish downloads for Windows, macOS and Linux (recommended)

The project already has a GitHub Actions workflow (`.github/workflows/build-desktop.yml`).
GitHub builds all three apps on its own machines, so you don't need a Windows or Mac.
It works on a free account if the repository is public.

**One-off test build** (no release): on GitHub open **Actions**, pick **Build desktop
apps**, click **Run workflow**. About 15 to 25 minutes later the three builds appear as
downloadable artifacts on that run.

**A proper release** that peers can download:
```bash
git tag v1.0.0
git push origin v1.0.0
```
The workflow builds all three and attaches them to a GitHub **Release** page with these
files:

| File | For |
|---|---|
| `AutoDS-windows-x64.zip` | Windows 10/11 |
| `AutoDS-macos-arm64.zip` | Apple Silicon Macs |
| `AutoDS-linux-x64.tar.gz` | Linux (built on Ubuntu 22.04 so it runs on many distros) |

Send peers the link to the Releases page plus `START-HERE.md`.

What peers should expect, so it doesn't surprise them:
- The apps are **not code-signed**. Windows shows SmartScreen and macOS asks for a
  right-click **Open**. `TROUBLESHOOTING.md` has the steps. Removing this needs paid
  developer certificates.
- There is no Intel-Mac build. Those users follow the source route.
- Each download is several hundred MB because it bundles Python and the ML libraries.
- I have not run this workflow myself, so do one test build before you tell anyone to
  rely on it.

## 2. Share the source code

```bash
python tools/make_share_zip.py /path/to/autods -o AutoDS-share.zip
```

This makes a small zip (about 1 MB) that leaves out `.venv`, `.git`, `.sessions`, `dist`,
caches, `.pkl`/`.log` files, anything over 25 MB, and generated `build/` output. It
**keeps** `build/AutoDS.spec`, which the desktop build needs. It also puts the run
scripts at the top of the project and the guide docs in `SHARE-GUIDE/`.

Peers unzip it and follow `START-HERE.md`, Way B.

## Keep these out of git and out of zips

| Folder | Why |
|---|---|
| `.sessions/` | Copies of every dataset anyone uploaded. Once committed, git keeps them in history forever. One project reached a 3 GB `.git` this way. |
| `.venv/`, `venv/` | 2 GB each and tied to your machine's Python. |
| `dist/` | PyInstaller output, several GB per build. |
| `*.pkl`, datasets, `.env` | Models and data are large, and may be private. |

`.gitignore` already lists these, but ignoring only affects files that were never
committed. Check that none are tracked:
```bash
git ls-files .sessions dist .venv | head
```
If that prints anything, run `git rm -r --cached .sessions` (and the same for the other
folders), then commit.

## Before you send anything

1. Run the tests: `python -m pytest tests/ -q`.
2. Make sure no private data is in the folder. Search for the datasets you tested with.
3. The Claude assistant's API key is stored in your user config folder, not in the
   project. Check there is no `.env` file and no key pasted into a source file:
   `grep -rniE "sk-ant|api_key *= *['\"]" backend desktop.py`.
4. Bump the version in your tag (`v1.0.1`, `v1.1.0`) for each release so peers can tell
   builds apart.
5. Tell peers which route to use: `START-HERE.md` is written for them.
