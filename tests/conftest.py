"""Shared pytest fixtures.

Without this, tests that construct a real backend.store.Session (e.g.
test_notebook_export.py, test_session_training_invalidation.py) persist
CSV files into the project's real .sessions/ directory on every run,
since Session.__init__ unconditionally calls _save_df_to_disk. That leaves
disk files behind permanently with nothing to clean them up mid-run, so
the folder grows every time the suite runs. Redirect the store's session
directory to a per-test-run temp folder so test data never touches the
real .sessions/ folder and is discarded automatically afterward.
"""
import pytest

from backend import store


@pytest.fixture(autouse=True)
def _isolated_session_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_SESSION_DIR", str(tmp_path))
    store.SESSIONS.clear()
    yield
    store.SESSIONS.clear()