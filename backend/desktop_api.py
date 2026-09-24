"""
Desktop-mode helper endpoints (used only by desktop.py).

The packaged app runs a private web server on 127.0.0.1 and shows the UI in
a browser window. These endpoints let that server notice when the window has
been closed (so it can quit by itself) and let the "Quit" button stop it.

They are inserted in front of the catch-all static-file mount by desktop.py,
so main.py does not need to change and the hosted version is unaffected (it
simply doesn't have these routes, and the page treats a 404 as "not desktop").
"""
import threading
import time

from fastapi import APIRouter

router = APIRouter(prefix="/api/desktop", tags=["desktop"])

# A page that hasn't checked in for this long is treated as gone even if it
# never said goodbye (browsers throttle hidden tabs to about one timer/minute).
CLIENT_STALE_SECONDS = 180

_lock = threading.Lock()
_clients: dict = {}          # page id -> last time it checked in
_seen_any = False            # has any page ever connected?
_shutdown_hook = None        # set by desktop.py


def set_shutdown_hook(fn) -> None:
    global _shutdown_hook
    _shutdown_hook = fn


def forget_all() -> None:
    """The native window process ended: don't wait for stale check-ins to expire."""
    with _lock:
        _clients.clear()


def snapshot() -> tuple:
    """(number of live pages, whether any page has ever connected)."""
    now = time.time()
    with _lock:
        for cid, seen in list(_clients.items()):
            if now - seen > CLIENT_STALE_SECONDS:
                del _clients[cid]
        return len(_clients), _seen_any


@router.get("/ping")
def ping(cid: str = ""):
    global _seen_any
    # No cid = just a probe (the launcher checking whether AutoDS is already
    # running); only real pages, which send an id, count as open windows.
    if cid:
        with _lock:
            _clients[cid] = time.time()
            _seen_any = True
    return {"desktop": True, "windows": snapshot()[0]}


@router.post("/bye")
def bye(cid: str = ""):
    with _lock:
        _clients.pop(cid, None)
    return {"ok": True}


@router.post("/quit")
def quit_app():
    if _shutdown_hook:
        # Small delay so this response reaches the browser before we go away.
        threading.Timer(0.4, _shutdown_hook).start()
    return {"ok": True}
