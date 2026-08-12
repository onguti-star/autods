"""
Simulation engine — Monte Carlo what-if scenario runner.

Given an outcome column and a driver column, runs bootstrap trials
to estimate the distribution of the outcome under a hypothetical change
to the driver. The driver's impact on the outcome is estimated from the
observed linear (OLS) relationship; results come with a method note
making clear this is correlation-based, not causal inference.
"""
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# ── Request model ────────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    outcome: str = Field(min_length=1, max_length=200)
    driver: str = Field(min_length=1, max_length=200)
    mode: Literal["percent", "fixed", "uniform_percent", "normal_percent"] = "percent"
    amount: float = 10.0
    low: float = -10.0
    high: float = 10.0
    trials: int = Field(default=1000, ge=100, le=10000)
    aggregate: Literal["sum", "mean", "median"] = "sum"


# ── Helpers ──────────────────────────────────────────────────────────────────

def aggregate_series(series: pd.Series, aggregate: str) -> float:
    """Return the requested aggregate statistic for a Series."""
    if aggregate == "mean":
        return float(series.mean())
    if aggregate == "median":
        return float(series.median())
    return float(series.sum())


def _compute_beta(x: pd.Series, y: pd.Series, driver: str, outcome: str):
    """
    Estimate the linear coefficient (beta) of x on y.

    Returns (beta, method_note).
    """
    if driver == outcome:
        return 1.0, "Scenario changes were applied directly to the outcome column."

    x_var = float(x.var(ddof=0))
    if x_var > 0:
        beta = float(x.cov(y) / x.var())
        note = (
            "Driver impact is estimated from the observed linear relationship "
            f"between '{driver}' and '{outcome}', not proven causality."
        )
    else:
        beta = 0.0
        note = f"'{driver}' has no variance, so estimated driver impact is zero."

    return beta, note


def _apply_delta(
    sample_x: pd.Series,
    mode: str,
    amount: float,
    low: float,
    high: float,
    n: int,
    seed: int,
) -> pd.Series:
    """Compute the per-row change to the driver column for one trial."""
    if mode == "percent":
        return sample_x * (amount / 100.0)

    if mode == "fixed":
        return pd.Series(amount, index=sample_x.index, dtype=float)

    rng = np.random.default_rng(seed)

    if mode == "uniform_percent":
        lo, hi = sorted([low, high])
        pct = pd.Series(rng.uniform(lo, hi, size=n)) / 100.0
        return sample_x * pct

    # normal_percent
    std = abs(high) if high else 5.0
    pct = pd.Series(rng.normal(amount, std, size=n)) / 100.0
    return sample_x * pct


# ── Core simulation function ─────────────────────────────────────────────────

def run_simulation(df: pd.DataFrame, req: SimulationRequest) -> dict:
    """
    Execute the Monte Carlo simulation and return the result payload.

    Raises ValueError for bad inputs so the caller can map to HTTP errors.
    """
    if req.outcome not in df.columns:
        raise ValueError(f"Outcome column '{req.outcome}' not found.")
    if req.driver not in df.columns:
        raise ValueError(f"Driver column '{req.driver}' not found.")
    if not pd.api.types.is_numeric_dtype(df[req.outcome]):
        raise ValueError(f"Outcome column '{req.outcome}' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df[req.driver]):
        raise ValueError(f"Driver column '{req.driver}' must be numeric.")

    sim_cols = [req.outcome] if req.outcome == req.driver else [req.outcome, req.driver]
    working = df[sim_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(working) < 5:
        raise ValueError("Simulation needs at least 5 complete numeric rows.")

    y = working[req.outcome].astype(float)
    x = y if req.driver == req.outcome else working[req.driver].astype(float)
    baseline = aggregate_series(y, req.aggregate)

    beta, method_note = _compute_beta(x, y, req.driver, req.outcome)

    # Bootstrap Monte Carlo
    n = len(working)
    base_y = y.reset_index(drop=True)
    base_x = x.reset_index(drop=True)
    seeds = pd.Series(range(req.trials)).sample(frac=1, random_state=None).index

    results: list[float] = []
    for seed in seeds:
        sampled_idx = (
            pd.Series(range(n))
            .sample(n=n, replace=True, random_state=int(seed))
            .to_numpy()
        )
        sample_y = base_y.iloc[sampled_idx].reset_index(drop=True)
        sample_x = base_x.iloc[sampled_idx].reset_index(drop=True)

        delta_x = _apply_delta(sample_x, req.mode, req.amount, req.low, req.high, n, int(seed))
        scenario_y = sample_y + (delta_x if req.driver == req.outcome else beta * delta_x)
        results.append(aggregate_series(scenario_y, req.aggregate))

    # Summarise distribution
    dist = pd.Series(results, dtype=float)
    q05, q50, q95 = dist.quantile([0.05, 0.5, 0.95]).tolist()
    min_v, max_v = float(dist.min()), float(dist.max())

    if min_v == max_v:
        labels = [f"{min_v:,.3g}"]
        counts = [len(dist)]
    else:
        bins = pd.cut(dist, bins=20)
        freq = bins.value_counts(sort=False)
        labels = [f"{iv.left:,.3g} to {iv.right:,.3g}" for iv in freq.index]
        counts = [int(v) for v in freq.values]

    mean_v = float(dist.mean())
    delta = mean_v - baseline
    pct_delta = (delta / baseline * 100.0) if baseline else None

    return {
        "outcome": req.outcome,
        "driver": req.driver,
        "mode": req.mode,
        "aggregate": req.aggregate,
        "trials": req.trials,
        "rows_used": int(n),
        "baseline": baseline,
        "mean": mean_v,
        "median": float(q50),
        "p05": float(q05),
        "p95": float(q95),
        "delta": float(delta),
        "pct_delta": None if pct_delta is None else float(pct_delta),
        "labels": labels,
        "counts": counts,
        "method_note": method_note,
    }