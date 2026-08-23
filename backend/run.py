
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

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[BACKEND_DIR],
    )