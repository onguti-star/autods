"""Regression tests: the pivot table's Copy button used to kill the desktop app.

Qt emits ``featurePermissionRequested`` the moment a page touches the
clipboard. pywebview 6.2's stock handler answers with a raw ``int``; PyQt6
rejects that with a TypeError -- and because PyQt6 aborts the process when an
exception escapes a Qt slot, the native window died with SIGABRT, which the
launcher's watchdog then read as "the window closed", shutting the whole app
down ("Window closed -- shutting down.").

``desktop._patch_qt_feature_permissions()`` replaces that handler with one
that answers with real PermissionPolicy enums and can never raise. These
tests pin that behaviour on both PyQt5 (test runner) and PyQt6 (the app's
own venv). They never construct QApplication/QWebEnginePage -- importing the
classes for their enums is enough, and the handler only calls
``self.setFeaturePermission`` -- so they run headless and without a working
GPU. The stock handler is deliberately never invoked through a live signal:
under PyQt6 that would abort the test process too.
"""
import pytest

pytest.importorskip("webview.platforms.qt")
pytest.importorskip("qtpy.QtWebEngineWidgets")

from qtpy.QtCore import QUrl
from qtpy.QtWebEngineWidgets import QWebEnginePage

import desktop
from webview.platforms import qt as wq


@pytest.fixture(scope="module", autouse=True)
def patched_handler():
    desktop._patch_qt_feature_permissions()
    web_page = getattr(getattr(wq, "BrowserView", None), "WebPage", None)
    handler = getattr(web_page, "onFeaturePermissionRequested", None)
    if handler is None:
        pytest.skip("this pywebview build has no WebPage.onFeaturePermissionRequested")
    if not hasattr(QWebEnginePage, "PermissionPolicy"):
        pytest.skip("this Qt build has no PermissionPolicy enums")
    # Must be OUR graft, not pywebview's stock (raw-int) handler.
    assert handler.__name__ == "on_feature_permission_requested"
    return handler


class _RecorderPage:
    """Duck-typed page that records permission answers instead of sending
    them to Qt -- the grafted handler only ever calls setFeaturePermission."""

    def __init__(self):
        self.calls = []

    def setFeaturePermission(self, url, feature, policy):
        self.calls.append((feature, policy))


def _features():
    """Every feature the app can plausibly be asked about, incl. clipboard
    when this Qt build knows it (PyQt5 doesn't)."""
    names = ["MediaVideoCapture", "MediaAudioVideoCapture",
             "Geolocation", "Notifications", "ClipboardReadWrite"]
    return [f for f in (getattr(QWebEnginePage.Feature, n, None) for n in names)
            if f is not None]


def test_answers_with_policy_enums_never_raw_ints(patched_handler):
    page = _RecorderPage()
    for feature in _features():
        patched_handler(page, QUrl("http://127.0.0.1/"), feature)

    assert page.calls, "the handler never answered at all"
    for feature, policy in page.calls:
        assert isinstance(policy, QWebEnginePage.PermissionPolicy), (
            f"raw-int policy {policy!r} for {feature.name} -- PyQt6 turns that "
            f"TypeError into a process abort"
        )


def test_grants_clipboard_and_media_denies_the_rest(patched_handler):
    page = _RecorderPage()
    granted = QWebEnginePage.PermissionPolicy.PermissionGrantedByUser
    denied = QWebEnginePage.PermissionPolicy.PermissionDeniedByUser

    patched_handler(page, QUrl("http://127.0.0.1/"),
                    QWebEnginePage.Feature.MediaVideoCapture)
    patched_handler(page, QUrl("http://127.0.0.1/"),
                    QWebEnginePage.Feature.Geolocation)
    patched_handler(page, QUrl("http://127.0.0.1/"),
                    QWebEnginePage.Feature.Notifications)
    clipboard = getattr(QWebEnginePage.Feature, "ClipboardReadWrite", None)
    if clipboard is not None:
        patched_handler(page, QUrl("http://127.0.0.1/"), clipboard)

    answers = dict(page.calls)
    assert answers[QWebEnginePage.Feature.MediaVideoCapture] == granted
    assert answers[QWebEnginePage.Feature.Geolocation] == denied
    assert answers[QWebEnginePage.Feature.Notifications] == denied
    if clipboard is not None:
        # This is the pivot Copy button's request -- it used to end in the
        # TypeError that took the window down.
        assert answers[clipboard] == granted


def test_permission_answer_never_raises(patched_handler):
    class _BrokenPage(_RecorderPage):
        def setFeaturePermission(self, url, feature, policy):
            raise RuntimeError("simulated Qt-side failure")

    # Must swallow the failure -- a raise escaping the slot aborts PyQt6.
    patched_handler(_BrokenPage(), QUrl("http://127.0.0.1/"),
                    QWebEnginePage.Feature.MediaVideoCapture)
