# AutoDS: sharing guide

AutoDS runs as its own desktop program. It starts a private server that only your
own computer can reach and shows the interface in a native window, so there is no
browser, address bar or tabs. Nothing is uploaded anywhere and it works offline.

There are two ways to get it running. Pick one.

## Way A: download the ready-made app (easiest, no Python needed)

Ask the person who shared AutoDS for the link to the **Releases** page and download
the file for your system:

| Your system | File | How to open it |
|---|---|---|
| Windows 10/11 | `AutoDS-windows-x64.zip` | Unzip, open the `AutoDS` folder, double-click `AutoDS.exe` |
| macOS (Apple Silicon) | `AutoDS-macos-arm64.zip` | Unzip, drag `AutoDS.app` to Applications, then right-click it and choose **Open** the first time |
| Linux | `AutoDS-linux-x64.tar.gz` | `tar xzf AutoDS-linux-x64.tar.gz`, then `cd AutoDS-linux && ./install.sh`, then open **AutoDS** from your applications menu |

The first launch shows a security warning on Windows and macOS. That is because
the app is not code-signed. The steps for getting past it are in
`TROUBLESHOOTING.md`.

## Way B: run it from the source code (works on any system with Python)

Use this if there is no download for your system (for example an Intel Mac), or you
want the latest code. You need **Python 3.12** (3.10 to 3.13 also work).

1. Unzip the project folder you were sent.
2. Find the helper for your system **in the project folder, next to `desktop.py`**:
   - Windows: `run-autods-windows.bat`
   - macOS: `run-autods-mac.command`
   - Linux: `run-autods-linux.sh`

   (If a helper is missing, copy it from this guide's `windows/`, `macos/` or `linux/`
   folder into the project folder.)
3. Run it. The first run creates a private environment and installs the packages
   (about 5 to 10 minutes and roughly 2 GB of disk). Later runs start in seconds.

Step-by-step instructions for each system:
`windows/INSTALL-WINDOWS.md`, `macos/INSTALL-MACOS.md`, `linux/INSTALL-LINUX.md`.

## If something goes wrong

Read `TROUBLESHOOTING.md`. It covers the problems people actually hit: the app
opening in a browser instead of its own window, "already running on port 8765",
a generic icon in the dock or taskbar, and the security warnings.

## For whoever is sharing the project

Read `FOR-THE-SHARER.md` before sending anything. It explains how to publish the
downloads for all three systems and which folders must not go into the zip or
into git.
