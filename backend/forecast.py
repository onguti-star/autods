"""
Trend & Forecast: the "plot historical counts, fit a trend, extrapolate N
periods forward" workflow (e.g. "content additions per year, projected 5
years out"). Mirrors the structure of recommender.py -- a detect() helper
plus a stateless run() that does everything in one call (no fitted model
needs to be cached between requests, unlike the recommender).
"""
import warnings as _warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

_warnings.filterwarnings("ignore")

VALID_PERIODS = ("year", "quarter", "month")
VALID_AGGS = ("count", "sum", "mean", "median", "min", "max")


# ── Detection ─────────────────────────────────────────────────────────────

def detect_forecast_setup(df: pd.DataFrame) -> dict:
    """Suggest a date/period column and a metric to trend & forecast."""
    cols = list(df.columns)

    date_col = None
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_col = c
            break

    if date_col is None:
        hints = ["date", "added", "created", "published", "timestamp", "time"]
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            if not any(h in c.lower() for h in hints):
                continue
            sample = df[c].dropna().astype(str).head(100)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True, format="mixed")
            if parsed.notna().mean() > 0.7:
                date_col = c
                break

    # An already-numeric "year" column (e.g. release_year, a year extracted
    # earlier in cleaning) is just as good a time axis as a real date.
    if date_col is None:
        for c in cols:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            vals = df[c].dropna()
            if vals.empty or vals.nunique() < 2:
                continue
            if vals.between(1600, 2200).mean() > 0.9 and "year" in c.lower():
                date_col = c
                break
        if date_col is None:
            for c in cols:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    continue
                vals = df[c].dropna()
                if vals.empty or vals.nunique() < 2:
                    continue
                if vals.between(1600, 2200).mean() > 0.95:
                    date_col = c
                    break

    if date_col is None:
        return {
            "ok": False,
            "best_guess": None,
            "message": "Couldn't find a date or year-like column to build a time trend from.",
        }

    metric_col = None
    for c in cols:
        if c == date_col or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        lc = c.lower()
        if any(h in lc for h in ["count", "sales", "revenue", "amount", "value", "total", "qty", "quantity", "price"]):
            metric_col = c
            break

    agg = "sum" if metric_col else "count"
    description = (
        f"Counting rows per year from '{date_col}'." if metric_col is None
        else f"Summing '{metric_col}' per year from '{date_col}'."
    )
    return {
        "ok": True,
        "best_guess": {
            "date_col": date_col,
            "period": "year",
            "metric_col": metric_col,
            "agg": agg,
            "description": description,
        },
    }


# ── Period resolution ────────────────────────────────────────────────────

def _resolve_periods(df: pd.DataFrame, date_col: str, period: str):
    """
    Return (idx, label) series for every row:
      idx   -- a numeric, strictly-increasing-by-1-per-period axis usable
               for regression (the year itself for "year"; a calendar
               ordinal for "quarter"/"month" so gaps in the data don't
               distort future extrapolation).
      label -- the matching display string ("2024", "2024Q2", "2024-02").
    """
    series = df[date_col]

    if period == "year" and pd.api.types.is_numeric_dtype(series):
        vals = series.dropna()
        if not vals.empty and vals.between(1600, 2200).mean() > 0.9:
            idx = pd.to_numeric(series, errors="coerce").round()
            return idx.astype(float), idx.astype("Int64").astype(str)

    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    if parsed.notna().sum() == 0:
        raise ValueError(
            f"Could not read '{date_col}' as dates (and it isn't a plain year column either)."
        )

    if period == "year":
        idx = parsed.dt.year.astype(float)
        label = parsed.dt.year.astype("Int64").astype(str)
    elif period == "quarter":
        per = parsed.dt.to_period("Q")
        idx = per.apply(lambda p: float(p.ordinal) if pd.notna(p) else np.nan)
        label = per.astype(str)
    elif period == "month":
        per = parsed.dt.to_period("M")
        idx = per.apply(lambda p: float(p.ordinal) if pd.notna(p) else np.nan)
        label = per.astype(str)
    else:
        raise ValueError(f"period must be one of {VALID_PERIODS}, got '{period}'.")

    return idx, label


def _future_labels(period: str, last_label: str, n: int) -> list[str]:
    if period == "year":
        start = int(last_label)
        return [str(start + i + 1) for i in range(n)]
    freq = "Q" if period == "quarter" else "M"
    p = pd.Period(last_label, freq=freq)
    return [str(p + i + 1) for i in range(n)]


# ── Aggregation ──────────────────────────────────────────────────────────

def aggregate_series(df: pd.DataFrame, date_col: str, period: str = "year",
                      metric_col: str | None = None, agg: str = "count") -> pd.DataFrame:
    """Bucket df[date_col] into periods and aggregate. Returns columns _idx, _label, _value, sorted by _idx."""
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in the dataset.")
    if period not in VALID_PERIODS:
        raise ValueError(f"period must be one of {VALID_PERIODS}, got '{period}'.")
    if agg not in VALID_AGGS:
        raise ValueError(f"agg must be one of {VALID_AGGS}, got '{agg}'.")

    if agg != "count":
        if not metric_col:
            raise ValueError("metric_col is required unless agg is 'count'.")
        if metric_col not in df.columns:
            raise ValueError(f"Column '{metric_col}' not found in the dataset.")
        if not pd.api.types.is_numeric_dtype(df[metric_col]) and not pd.api.types.is_numeric_dtype(
            pd.to_numeric(df[metric_col], errors="coerce")
        ):
            pass  # fall through; the to_numeric below will just produce NaNs and get caught by the empty check

    idx, label = _resolve_periods(df, date_col, period)
    work = pd.DataFrame({"_idx": idx, "_label": label})
    if agg != "count":
        work["_metric"] = pd.to_numeric(df[metric_col], errors="coerce")
        work = work.dropna(subset=["_idx", "_metric"])
    else:
        work = work.dropna(subset=["_idx"])

    if work.empty:
        raise ValueError(
            "No usable rows after parsing -- check that the date column actually holds dates/years"
            + ("" if agg == "count" else " and the metric column holds numbers.")
        )

    if agg == "count":
        grouped = work.groupby(["_idx", "_label"], as_index=False).size().rename(columns={"size": "_value"})
    else:
        grouped = work.groupby(["_idx", "_label"], as_index=False)["_metric"].agg(agg).rename(columns={"_metric": "_value"})

    grouped = grouped.sort_values("_idx").reset_index(drop=True)
    if len(grouped) < 2:
        raise ValueError(
            "Need at least 2 distinct time periods to fit a trend -- try a finer period (e.g. month instead of year)."
        )
    return grouped


# ── Fit + forecast ───────────────────────────────────────────────────────

def _format_equation(coeffs: np.ndarray, degree: int) -> str:
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        c = round(float(c), 4)
        if power == 0:
            terms.append(f"{c:+.4g}")
        elif power == 1:
            terms.append(f"{c:+.4g}·x")
        else:
            terms.append(f"{c:+.4g}·x^{power}")
    return "y ≈ " + " ".join(terms).lstrip("+").strip()


def run_forecast(df: pd.DataFrame, date_col: str, period: str = "year",
                  metric_col: str | None = None, agg: str = "count",
                  degree: int = 1, periods_ahead: int = 5) -> dict:
    """
    Aggregate df[date_col] into periods, fit a degree-1 (linear) or
    degree-2 (quadratic) trend, and extrapolate periods_ahead steps beyond
    the last observed period. Returns JSON-safe historical/trend/future
    series ready to chart, plus fit diagnostics and any caveats.
    """
    grouped = aggregate_series(df, date_col, period, metric_col, agg)
    n = len(grouped)
    x = grouped["_idx"].to_numpy(dtype=float)
    y = grouped["_value"].to_numpy(dtype=float)
    labels = grouped["_label"].astype(str).tolist()

    if np.ptp(x) == 0:
        raise ValueError("All rows fall into the same time period -- nothing to fit a trend over.")

    requested_degree = max(1, min(int(degree), 2))
    effective_degree = requested_degree
    warnings_out = []
    if n < effective_degree + 2 and effective_degree > 1:
        warnings_out.append(
            f"Only {n} data points -- using a linear trend instead of a degree-{effective_degree} curve (too few points to fit reliably)."
        )
        effective_degree = 1

    coeffs = np.polyfit(x, y, effective_degree)
    poly = np.poly1d(coeffs)
    fitted = poly(x)
    r2 = float(r2_score(y, fitted)) if n >= 2 else None

    periods_ahead = max(1, min(int(periods_ahead), 50))
    last_x = float(x[-1])
    future_x = np.array([last_x + i + 1 for i in range(periods_ahead)])
    future_y = poly(future_x)
    future_labels = _future_labels(period, labels[-1], periods_ahead)

    if agg == "count":
        display_hist = np.round(y)
        display_fit = np.round(fitted)
        display_future = np.clip(np.round(future_y), 0, None)
    else:
        display_hist = y
        display_fit = fitted
        display_future = future_y

    if n < 4:
        warnings_out.append(f"Only {n} historical data points -- treat this forecast as a rough extrapolation.")
    if r2 is not None and r2 < 0.3:
        warnings_out.append(f"The trend only explains {round(r2 * 100)}% of the variation (R² = {round(r2, 3)}) -- the data may not follow a steady trend.")

    return {
        "date_col": date_col,
        "period": period,
        "metric_col": metric_col,
        "agg": agg,
        "degree": effective_degree,
        "requested_degree": requested_degree,
        "n_points": int(n),
        "r2": None if r2 is None else round(r2, 4),
        "equation": _format_equation(coeffs, effective_degree),
        "historical": {"labels": labels, "values": [round(float(v), 4) for v in display_hist]},
        "trend": {"labels": labels, "values": [round(float(v), 4) for v in display_fit]},
        "future": {"labels": future_labels, "values": [round(float(v), 4) for v in display_future]},
        "warnings": warnings_out,
    }
