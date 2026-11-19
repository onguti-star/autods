"""Runtime configuration shared by the local and container launchers."""

from __future__ import annotations

import os


def env_value(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def env_port(name: str, default: int) -> int:
    """Return a valid TCP port, falling back instead of crashing on bad env input."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        port = int(raw)
    except ValueError:
        return default
    return port if 1 <= port <= 65535 else default


SERVER_HOST = env_value("AUTODS_HOST", "0.0.0.0")
# Render and several other hosting platforms provide PORT rather than the
# app-specific AUTODS_PORT. Prefer the explicit app setting, then the hosting
# platform setting, then the local development default.
SERVER_PORT = env_port("AUTODS_PORT", env_port("PORT", 8000))
