"""
Tests for backend/simulation.py

Covers: all four scenario modes, driver==outcome edge case,
zero-variance driver, error paths, and aggregate options.
Follows the same style as the rest of the test suite (no pytest imports).
"""
import numpy as np
import pandas as pd

from backend.simulation import SimulationRequest, aggregate_series, run_simulation


# ── Helpers ──────────────────────────────────────────────────────────────────

def _df():
    """Small deterministic DataFrame with a clear linear relationship."""
    rng = np.random.default_rng(42)
    n = 50
    x = rng.uniform(10, 100, size=n)
    y = 3.0 * x + rng.normal(0, 2, size=n)
    return pd.DataFrame({"revenue": y, "units": x})


def _req(**kwargs) -> SimulationRequest:
    defaults = dict(outcome="revenue", driver="units", mode="percent",
                    amount=10.0, low=-10.0, high=10.0, trials=200, aggregate="sum")
    defaults.update(kwargs)
    return SimulationRequest(**defaults)


def _raises(fn, exc_type, substring):
    """Assert fn() raises exc_type whose message contains substring."""
    try:
        fn()
        raise AssertionError(f"Expected {exc_type.__name__} containing '{substring}'")
    except exc_type as e:
        assert substring in str(e), f"Expected '{substring}' in: {e}"


# ── aggregate_series ──────────────────────────────────────────────────────────

def test_aggregate_sum():
    assert aggregate_series(pd.Series([1.0, 2.0, 3.0]), "sum") == 6.0


def test_aggregate_mean():
    assert aggregate_series(pd.Series([1.0, 2.0, 3.0]), "mean") == 2.0


def test_aggregate_median():
    assert aggregate_series(pd.Series([1.0, 2.0, 10.0]), "median") == 2.0


def test_aggregate_defaults_to_sum():
    assert aggregate_series(pd.Series([4.0, 5.0]), "anything_else") == 9.0


# ── percent mode ─────────────────────────────────────────────────────────────

def test_percent_mode_positive_delta():
    result = run_simulation(_df(), _req(mode="percent", amount=10.0))
    assert result["delta"] > 0
    assert result["rows_used"] == 50
    assert result["trials"] == 200


def test_percent_mode_negative_amount():
    result = run_simulation(_df(), _req(mode="percent", amount=-20.0))
    assert result["delta"] < 0


# ── fixed mode ───────────────────────────────────────────────────────────────

def test_fixed_mode_returns_result():
    result = run_simulation(_df(), _req(mode="fixed", amount=5.0))
    assert result["mean"] != result["baseline"]
    assert result["mode"] == "fixed"


# ── uniform_percent mode ──────────────────────────────────────────────────────

def test_uniform_percent_mode():
    result = run_simulation(_df(), _req(mode="uniform_percent", low=-5.0, high=5.0))
    assert abs(result["pct_delta"]) < 20


# ── normal_percent mode ───────────────────────────────────────────────────────

def test_normal_percent_mode():
    result = run_simulation(_df(), _req(mode="normal_percent", amount=0.0, high=5.0))
    assert result["pct_delta"] is not None
    assert len(result["labels"]) > 0


# ── driver == outcome ─────────────────────────────────────────────────────────

def test_driver_equals_outcome():
    result = run_simulation(_df(), _req(outcome="revenue", driver="revenue", mode="percent", amount=10.0))
    assert "directly to the outcome column" in result["method_note"]
    assert result["delta"] > 0


# ── zero-variance driver ──────────────────────────────────────────────────────

def test_zero_variance_driver():
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 150.0, 130.0, 170.0],
        "constant": [5.0, 5.0, 5.0, 5.0, 5.0],
    })
    result = run_simulation(df, _req(outcome="revenue", driver="constant"))
    assert "no variance" in result["method_note"]
    assert abs(result["delta"]) < 20


# ── aggregate options ─────────────────────────────────────────────────────────

def test_mean_aggregate():
    df = _df()
    result = run_simulation(df, _req(aggregate="mean"))
    assert result["aggregate"] == "mean"
    assert abs(result["baseline"] - float(df["revenue"].mean())) < 0.01


def test_median_aggregate():
    result = run_simulation(_df(), _req(aggregate="median"))
    assert result["aggregate"] == "median"


# ── response shape ────────────────────────────────────────────────────────────

def test_result_keys_present():
    result = run_simulation(_df(), _req())
    required = {
        "outcome", "driver", "mode", "aggregate", "trials", "rows_used",
        "baseline", "mean", "median", "p05", "p95", "delta", "pct_delta",
        "labels", "counts", "method_note",
    }
    assert required <= result.keys()


def test_labels_and_counts_same_length():
    result = run_simulation(_df(), _req())
    assert len(result["labels"]) == len(result["counts"])


def test_p05_le_median_le_p95():
    result = run_simulation(_df(), _req())
    assert result["p05"] <= result["median"] <= result["p95"]


# ── error paths ───────────────────────────────────────────────────────────────

def test_outcome_column_not_found():
    _raises(lambda: run_simulation(_df(), _req(outcome="missing")),
            ValueError, "Outcome column 'missing' not found")


def test_driver_column_not_found():
    _raises(lambda: run_simulation(_df(), _req(driver="missing")),
            ValueError, "Driver column 'missing' not found")


def test_non_numeric_outcome():
    df = pd.DataFrame({"label": ["a","b","c","d","e"], "units": [1.0,2.0,3.0,4.0,5.0]})
    _raises(lambda: run_simulation(df, _req(outcome="label", driver="units")),
            ValueError, "must be numeric")


def test_non_numeric_driver():
    df = pd.DataFrame({"revenue": [1.0,2.0,3.0,4.0,5.0], "label": ["a","b","c","d","e"]})
    _raises(lambda: run_simulation(df, _req(outcome="revenue", driver="label")),
            ValueError, "must be numeric")


def test_too_few_rows():
    df = pd.DataFrame({"revenue": [1.0, 2.0], "units": [3.0, 4.0]})
    _raises(lambda: run_simulation(df, _req()),
            ValueError, "at least 5")


def test_missing_rows_are_dropped():
    df = pd.DataFrame({
        "revenue": [1.0, None, None, None, None],
        "units":   [2.0, None, None, None, None],
    })
    _raises(lambda: run_simulation(df, _req()),
            ValueError, "at least 5")


# ── pct_delta edge: zero baseline ────────────────────────────────────────────

def test_pct_delta_is_none_when_baseline_zero():
    df = pd.DataFrame({
        "revenue": [0.0, 0.0, 0.0, 0.0, 0.0],
        "units":   [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    result = run_simulation(df, _req(outcome="revenue", driver="units"))
    assert result["pct_delta"] is None