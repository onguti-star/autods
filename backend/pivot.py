"""Pivot tables: group rows by one or more fields, optionally spread another
field across the columns, and aggregate numeric values into the cells.

The result is returned as plain JSON (headers + rows) so the frontend can
render it as a table, and so it can be dropped into the HTML report or the
notebook export without going through pandas again.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

AGGREGATES = ("sum", "mean", "median", "min", "max", "count", "nunique", "std")
DATE_BUCKETS = ("year", "quarter", "month", "weekday")
PERCENT_MODES = ("none", "of_total", "of_row", "of_col")

MAX_ROWS = 500          # rows shown; the rest are summarised in a note
MAX_COLS = 40           # columns shown
MAX_FIELD_LEVELS = 3    # how many fields may be nested on the row side


class PivotRequest(BaseModel):
    rows: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    aggregate: Literal["sum", "mean", "median", "min", "max", "count", "nunique", "std"] = "sum"
    totals: bool = True
    percent: Literal["none", "of_total", "of_row", "of_col"] = "none"
    date_by: Optional[Literal["year", "quarter", "month", "weekday"]] = None
    sort_by: Literal["label", "value"] = "label"
    descending: bool = False
    fill_zero: bool = True
    decimals: int = Field(default=2, ge=0, le=6)


def _bucket_dates(series: pd.Series, bucket: str) -> pd.Series | None:
    """Return the series grouped by year/quarter/month/weekday, or None if the
    column does not look like dates (so numbers and plain text are left alone)."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    else:
        non_null = series.dropna()
        if non_null.empty:
            return None
        sample = non_null.astype(str).head(200)
        trial = pd.to_datetime(sample, dayfirst=True, format="mixed", errors="coerce")
        if trial.notna().mean() < 0.9:
            return None
        parsed = pd.to_datetime(series.astype("string"), dayfirst=True, format="mixed", errors="coerce")
    if bucket == "year":
        out = parsed.dt.year.astype("Int64").astype("string")
    elif bucket == "quarter":
        out = parsed.dt.year.astype("Int64").astype("string") + "-Q" + parsed.dt.quarter.astype("Int64").astype("string")
    elif bucket == "month":
        out = parsed.dt.strftime("%Y-%m")
    else:  # weekday
        out = parsed.dt.day_name()
    return out


def _label(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "(blank)"
    return str(value)


def _clean_number(x, decimals: int):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if np.isinf(x):
            return None
        return round(float(x), decimals)
    return x


def build_pivot(df: pd.DataFrame, req: PivotRequest) -> dict:
    rows = [c for c in req.rows if c]
    columns = [c for c in req.columns if c]
    values = [c for c in req.values if c]

    if not rows and not columns:
        raise ValueError("Pick at least one field for Rows (or Columns) to group by.")
    if len(rows) > MAX_FIELD_LEVELS:
        raise ValueError(f"Use at most {MAX_FIELD_LEVELS} fields in Rows.")
    if len(columns) > 2:
        raise ValueError("Use at most 2 fields in Columns.")

    used = rows + columns + values
    missing = [c for c in used if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {', '.join(missing)}.")
    if set(rows) & set(columns):
        raise ValueError("A field can't be in both Rows and Columns.")
    if set(values) & (set(rows) | set(columns)):
        raise ValueError("A field can't be both a value and a grouping field. Use a different one for Values.")

    notes: list[str] = []
    agg = req.aggregate

    # Numeric aggregates need numeric values; count/nunique work on anything.
    if agg not in ("count", "nunique"):
        for v in values:
            if not pd.api.types.is_numeric_dtype(df[v]):
                raise ValueError(
                    f"'{v}' isn't numeric, so it can't be summed or averaged. "
                    "Pick a numeric column, or switch the calculation to Count."
                )

    work = df[used].copy() if used else df.copy()

    # Optional date bucketing on grouping fields that actually hold dates
    if req.date_by:
        bucketed = []
        for field in rows + columns:
            grouped = _bucket_dates(work[field], req.date_by)
            if grouped is not None:
                work[field] = grouped
                bucketed.append(field)
        if bucketed:
            notes.append(f"Dates in {', '.join(bucketed)} grouped by {req.date_by}.")
        else:
            notes.append(f"No date columns among your Rows/Columns, so \"group dates by {req.date_by}\" was skipped.")

    # With no value column the pivot simply counts rows.
    counting_rows = not values
    if counting_rows:
        work["__rows__"] = 1
        values = ["__rows__"]
        agg_used = "sum"
        value_title = "Rows"
    else:
        agg_used = agg
        value_title = None

    # Percentages only make sense for additive measures.
    percent = req.percent
    if percent != "none":
        if not counting_rows and agg not in ("sum", "count"):
            raise ValueError("Percentages work with Sum or Count only. Switch the calculation, or turn percentages off.")
        if len(values) != 1:
            raise ValueError("Percentages work with one value at a time. Keep a single Values field.")
        if percent in ("of_row", "of_col") and not columns:
            raise ValueError("\"% of row\" and \"% of column\" need a field in Columns.")

    # Treat blanks in grouping fields as their own group instead of dropping rows.
    for field in rows + columns:
        if work[field].isna().any():
            work[field] = work[field].astype("object").where(work[field].notna(), "(blank)")

    margins = bool(req.totals and rows)
    if req.totals and not rows:
        notes.append("Totals are shown when at least one field is in Rows.")
    try:
        table = pd.pivot_table(
            work,
            index=rows or None,
            columns=columns or None,
            values=values,
            aggfunc=agg_used,
            margins=margins,
            margins_name="Total",
            observed=True,
            dropna=False,
        )
    except Exception as exc:  # pandas raises assorted errors for odd inputs
        raise ValueError(f"Couldn't build that pivot: {exc}")

    if table.empty:
        raise ValueError("That combination has no data. Try different fields.")

    # pivot_table can return a Series for a single value; normalise to a DataFrame
    if isinstance(table, pd.Series):
        table = table.to_frame()

    # Keep the value columns in the order the person picked them, and drop the
    # redundant value-name header when there is only one value.
    if isinstance(table.columns, pd.MultiIndex):
        table = table.reindex(columns=values, level=0)
        if len(values) == 1:
            table.columns = table.columns.droplevel(0)
    elif len(values) > 1:
        table = table[[v for v in values if v in table.columns]]

    # Percentages: divide by the matching total, then express as a percent.
    if percent != "none":
        base = table.copy()
        if margins:
            full = table
        else:
            full = pd.pivot_table(
                work, index=rows or None, columns=columns or None, values=values,
                aggfunc=agg_used, margins=True, margins_name="Total", observed=True, dropna=False,
            )
            if isinstance(full, pd.Series):
                full = full.to_frame()
            if isinstance(full.columns, pd.MultiIndex) and len(values) == 1:
                full.columns = full.columns.droplevel(0)
        has_total_col = bool(columns) and "Total" in list(full.columns)
        grand = float(full.iloc[-1, -1])
        if margins:
            base = base.iloc[:-1, :]
            base = base.drop(columns=["Total"], errors="ignore")
            full_body = full.iloc[:-1, :]
        else:
            full_body = full.iloc[:-1, :]
        totals_row = full.iloc[-1]
        if percent == "of_total":
            pct = base / grand * 100 if grand else base * np.nan
        elif percent == "of_row":
            denom = full_body["Total"] if has_total_col else full_body.sum(axis=1)
            pct = base.div(denom.reindex(base.index).replace(0, np.nan), axis=0) * 100
        else:  # of_col
            denom = totals_row.reindex(base.columns)
            pct = base.div(denom.replace(0, np.nan), axis=1) * 100
        if margins:
            # rebuild the Total row/column as percentages of the same base
            if percent == "of_total":
                pct["Total"] = pct.sum(axis=1)
                pct.loc["Total"] = pct.sum(axis=0)
            elif percent == "of_row":
                pct["Total"] = 100.0
                pct.loc["Total"] = (base.sum(axis=0) / grand * 100) if grand else np.nan
                pct.loc["Total", "Total"] = 100.0
            else:
                pct.loc["Total"] = 100.0
                pct["Total"] = (base.sum(axis=1) / grand * 100) if grand else np.nan
                pct.loc["Total", "Total"] = 100.0
        table = pct

    if req.fill_zero:
        table = table.fillna(0)

    # Sorting (the Total row, when present, is always last and stays last)
    total_row = None
    body = table
    if margins and len(table) > 1:
        total_row = table.iloc[[-1]]
        body = table.iloc[:-1]
    if req.sort_by == "value" and len(body) > 1:
        key = body.iloc[:, -1].astype(float).fillna(-np.inf).to_numpy()
        order = np.argsort(key, kind="stable")
        if req.descending:
            order = order[::-1]
        body = body.iloc[order]
    elif req.descending:
        body = body.sort_index(ascending=False)
    table = pd.concat([body, total_row]) if total_row is not None else body
    total_row_present = total_row is not None

    # Size limits
    total_rows_before = len(table)
    if total_rows_before > MAX_ROWS + (1 if total_row_present else 0):
        keep = table.iloc[:MAX_ROWS]
        if total_row_present:
            keep = pd.concat([keep, table.loc[["Total"]]])
        notes.append(f"Showing the first {MAX_ROWS:,} of {total_rows_before - (1 if total_row_present else 0):,} rows. Add a filter or group by fewer fields to see the rest.")
        table = keep
    if table.shape[1] > MAX_COLS:
        notes.append(f"Showing the first {MAX_COLS} of {table.shape[1]} columns.")
        table = table.iloc[:, :MAX_COLS]

    # Flatten headers
    row_fields = list(rows) if rows else [""]
    if isinstance(table.columns, pd.MultiIndex):
        header = [" · ".join(_label(p) for p in col if p != "") for col in table.columns]
    else:
        header = [_label(c) for c in table.columns]
    if counting_rows:
        header = ["Rows" if h == "__rows__" else h for h in header]
    elif not columns:
        header = [f"{agg} of {h}" if h in values else h for h in header]

    out_rows = []
    for idx, rec in zip(table.index, table.itertuples(index=False)):
        labels = list(idx) if isinstance(idx, tuple) else [idx]
        out_rows.append({
            "labels": [_label(x) for x in labels],
            "cells": [_clean_number(v, req.decimals) for v in rec],
            "is_total": _label(labels[0]) == "Total" and bool(margins),
        })

    if not rows:
        # Only column fields: a single summary row
        for r in out_rows:
            r["labels"] = ["All"]

    # Summary caption
    if counting_rows:
        what = "Number of rows"
    else:
        verb = {"sum": "Sum", "mean": "Average", "median": "Median", "min": "Minimum", "max": "Maximum",
                "count": "Count", "nunique": "Distinct count", "std": "Std. deviation"}[agg]
        what = f"{verb} of {', '.join(values)}"
    caption = what
    if rows:
        caption += f" by {', '.join(rows)}"
    if columns:
        caption += f", across {', '.join(columns)}"
    if percent != "none":
        caption += {"of_total": " (as % of grand total)", "of_row": " (as % of each row)", "of_col": " (as % of each column)"}[percent]

    return {
        "caption": caption,
        "row_fields": row_fields,
        "column_fields": columns,
        "headers": header,
        "rows": out_rows,
        "n_rows": len(out_rows),
        "is_percent": percent != "none",
        "notes": notes,
        "source_rows": int(len(df)),
        "value_label": value_title or agg,
    }
