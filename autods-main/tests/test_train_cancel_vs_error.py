"""
Regression tests for a real bug: when the training subprocess died without
putting a result on the queue, the /api/train/{id}/status endpoint always
reported status "cancelled" -- even if the user never clicked Cancel. On large
datasets the most likely reason a training process disappears without a trace
is the OS killing it for using too much memory (or a native crash), not a user
cancellation, and telling the user "Training cancelled" in that case is
actively misleading (it reads as "you did this" instead of "training failed").

These tests exercise backend.main.train_status directly against a fake job
entry so they run fast and deterministically -- no real subprocess needed.
"""
import queue as queue_module

import numpy as np
import pandas as pd
import pytest

from backend import main, store


class _FakeProcess:
    def __init__(self, exitcode):
        self.exitcode = exitcode

    def is_alive(self):
        return False


class _FakeManager:
    def shutdown(self):
        pass


class _EmptyQueue:
    def get_nowait(self):
        raise queue_module.Empty()


def _make_session():
    df = pd.DataFrame({"target": [0, 1] * 10, "x": np.arange(20)})
    return store.create_session(df, "test.csv")


def _install_fake_job(session_id, *, exitcode, cancel_requested, n_rows=None):
    main.TRAIN_JOBS[session_id] = {
        "process": _FakeProcess(exitcode),
        "manager": _FakeManager(),
        "progress": [],
        "result_queue": _EmptyQueue(),
        "target": "target",
        "cancel_requested": cancel_requested,
        "n_rows": n_rows,
    }


def test_process_killed_without_explicit_cancel_is_reported_as_error_not_cancelled():
    session = _make_session()
    _install_fake_job(session.id, exitcode=-9, cancel_requested=False, n_rows=500_000)

    result = main.train_status(session.id)

    assert result["status"] == "error"
    assert "memory" in result["error"].lower()
    assert session.id not in main.TRAIN_JOBS  # job was cleaned up


def test_explicit_user_cancel_is_still_reported_as_cancelled():
    session = _make_session()
    _install_fake_job(session.id, exitcode=-9, cancel_requested=True)

    result = main.train_status(session.id)

    assert result["status"] == "cancelled"


def test_unexplained_process_death_without_signal_is_still_an_error():
    session = _make_session()
    _install_fake_job(session.id, exitcode=0, cancel_requested=False)

    result = main.train_status(session.id)

    assert result["status"] == "error"
    assert "cancel" not in result["error"].lower()


def test_cancel_endpoint_marks_job_as_explicitly_cancelled_before_cleanup():
    session = _make_session()
    _install_fake_job(session.id, exitcode=None, cancel_requested=False)
    job_ref = main.TRAIN_JOBS[session.id]

    main.cancel_train(session.id)

    assert job_ref["cancel_requested"] is True
    assert session.id not in main.TRAIN_JOBS
