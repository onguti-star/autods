
import os
import sys

# main.py uses relative imports (e.g. "from . import assistant"), which only
# work when it's loaded as part of the "backend" package rather than as a
# bare top-level "main" module. So we point uvicorn at "backend.main:app"
# and make sure the project root (the parent of this backend/ folder) is on
# the path — including for uvicorn's --reload subprocess, which starts a
# fresh interpreter that won't inherit sys.path changes, only env vars.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import uvicorn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dev", action="store_true", help="Enable hot-reload (dev only — kills training mid-run if backend files change)")
args = parser.parse_args()

if __name__ == "__main__":
    if args.dev:
        # --reload starts uvicorn's own subprocess from the "backend.main:app"
        # string below, so the router patch a few lines down (which only runs
        # in *this* process) wouldn't reach it. The optional Claude assistant
        # is skipped in --dev mode as a result; restart without --dev to use it.
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[BACKEND_DIR],
        )
    else:
        from backend.main import app

        # Optional, additional online assistant (Claude) -- adds its own
        # /api/claude/* routes if backend/claude_assistant.py is present.
        # Needs no setup and does nothing until an API key is saved in the
        # app; everything else about AutoDS works exactly the same without it.
        try:
            from backend import claude_assistant
            if not any(getattr(r, "path", "").startswith("/api/claude") for r in app.router.routes):
                app.router.routes[:0] = list(claude_assistant.router.routes)
        except Exception as exc:
            print(f"Claude assistant not loaded: {exc}")

        uvicorn.run(app, host="0.0.0.0", port=8000)