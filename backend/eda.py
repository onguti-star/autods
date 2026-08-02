"""
Automated EDA: produces a JSON-serializable profile of a dataframe
without the user writing any pandas code.
"""
import numpy as np
import pandas as pd

from . import nlp

PROFILE_SAMPLE_ROWS = 10_000
MAX_CORRELATION_COLUMNS = 60
TARGET_NAME_HINTS = {
    "target",
    "label",
    "class",
    "category",
    "outcome",
    "result",
    "status",
    "type",
    "score",
    "rating",
    "price",
    "amount",
    "cost",
    "sales",
    "revenue",
    "profit",
    "income",
    "churn",
    "default",
    "fraud",
    "risk",
    "approved",
    "passed",
    "survived",
}


def _safe_json(val):
    if val is pd.NA:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (pd.Timestamp,)):
        return val.isoformat()
    if pd.isna(val):
        return None
    return val


def safe_preview(df: pd.DataFrame, n: int = 10) -> list:
    """df.head(n) as JSON-safe records (NaN/NaT -> None)."""
    records = df.head(n).to_dict(orient="records")
    return [{k: _safe_json(v) for k, v in row.items()} for row in records]


def column_profile(df: pd.DataFrame, col: str, sample_df: pd.DataFrame | None = None) -> dict:
    s_full = df[col]
    s = sample_df[col] if sample_df is not None else s_full
    n = len(s_full)
    missing = int(s_full.isna().sum())
    profile = {
        "name": col,
        "dtype": str(s_full.dtype),
        "missing": missing,
        "missing_pct": round(missing / n * 100, 2) if n else 0,
        "unique": int(s_full.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(s_full) and not pd.api.types.is_bool_dtype(s_full):
        desc = s_full.describe()
        profile["type"] = "numeric"
        profile["stats"] = {k: _safe_json(v) for k, v in desc.items()}
        # histogram (10 bins)
        clean = s.dropna()
        if len(clean) > 0:
            counts, edges = np.histogram(clean, bins=min(10, max(1, clean.nunique())))
            profile["histogram"] = {
                "counts": [int(c) for c in counts],
                "edges": [round(float(e), 4) for e in edges],
            }
        else:
            profile["histogram"] = {"counts": [], "edges": []}
    else:
        if nlp.is_text_column(s):
            profile["type"] = "text"
            profile["text_stats"] = nlp.text_column_stats(s)
        else:
            profile["type"] = "categorical"
            top = s.value_counts(dropna=True).head(10)
            profile["top_values"] = [
                {"value": str(idx), "count": int(cnt)} for idx, cnt in top.items()
            ]

    return profile


def correlation_matrix(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {"columns": list(numeric_df.columns), "matrix": []}
    if len(numeric_df) > PROFILE_SAMPLE_ROWS:
        numeric_df = numeric_df.sample(PROFILE_SAMPLE_ROWS, random_state=42)
    if numeric_df.shape[1] > MAX_CORRELATION_COLUMNS:
        numeric_df = numeric_df.iloc[:, :MAX_CORRELATION_COLUMNS]
    corr = numeric_df.corr(numeric_only=True).round(3)
    corr = corr.fillna(0)
    return {
        "columns": list(corr.columns),
        "matrix": corr.values.tolist(),
    }


def describe_dataframe(df: pd.DataFrame) -> dict:
    """Return a JSON-safe pandas describe(include='all') summary."""
    if df.empty or df.shape[1] == 0:
        return {"index": [], "columns": [], "data": []}

    described = df.describe(include="all").replace({np.nan: None})
    return {
        "index": [str(idx) for idx in described.index],
        "columns": [str(col) for col in described.columns],
        "data": [
            [_safe_json(value) for value in row]
            for row in described.to_numpy(dtype=object).tolist()
        ],
    }


def _target_name_score(col: str) -> int:
    normalized = col.lower().replace("-", "_").replace(" ", "_")
    parts = {p for p in normalized.split("_") if p}
    if normalized in TARGET_NAME_HINTS:
        return 35
    if parts & TARGET_NAME_HINTS:
        return 24
    if normalized.endswith(("_id", "id")):
        return -30
    return 0


def learning_recommendation(df: pd.DataFrame) -> dict:
    """Suggest whether the dataset is more likely supervised or unsupervised.

    This is a heuristic: supervised learning requires a human-confirmed target
    column, but common label-like names and healthy target cardinality are useful
    signals.
    """
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    candidates = []

    for col in df.columns:
        s = df[col]
        non_missing = int(s.notna().sum())
        if n_rows == 0 or non_missing == 0:
            continue

        unique = int(s.nunique(dropna=True))
        unique_ratio = unique / max(non_missing, 1)
        is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        is_bool = pd.api.types.is_bool_dtype(s)
        is_datetime = pd.api.types.is_datetime64_any_dtype(s)

        score = _target_name_score(col)
        reasons = []
        if score > 0:
            reasons.append("name looks like a target/label")

        if is_datetime:
            score -= 35
            reasons.append("date/time columns are rarely prediction targets")
        elif is_bool:
            score += 30
            reasons.append("binary values fit classification")
        elif is_numeric:
            if 2 <= unique <= max(20, int(non_missing * 0.2)):
                score += 18
                reasons.append("numeric values could be classes or scores")
            elif unique_ratio < 0.95:
                score += 12
                reasons.append("numeric values are suitable for regression")
            else:
                score -= 12
                reasons.append("almost every value is unique")
        else:
            if 2 <= unique <= min(50, max(2, int(non_missing * 0.5))):
                score += 28
                reasons.append("limited categories fit classification")
            elif unique_ratio > 0.8:
                score -= 20
                reasons.append("mostly unique text is unlikely to be a target")

        if unique <= 1:
            score -= 40
            reasons.append("constant columns cannot be useful targets")
        if unique_ratio > 0.98 and not is_numeric:
            score -= 20
        if str(col).lower() in {"id", "uuid", "index"} or str(col).lower().endswith(("_id", " id")):
            score -= 35
            reasons.append("identifier-like columns are usually not targets")

        if score >= 20:
            problem_type = "regression" if is_numeric and unique > max(20, int(non_missing * 0.2)) else "classification"
            candidates.append({
                "column": col,
                "score": int(score),
                "problem_type": problem_type,
                "unique": unique,
                "missing_pct": round((n_rows - non_missing) / n_rows * 100, 2) if n_rows else 0,
                "reasons": reasons[:3],
            })

    candidates.sort(key=lambda item: item["score"], reverse=True)

    if candidates:
        best = candidates[0]
        return {
            "mode": "supervised",
            "confidence": "high" if best["score"] >= 55 else "medium",
            "summary": (
                f"This dataset looks suitable for supervised learning if `{best['column']}` "
                f"is the value you want to predict."
            ),
            "recommended_next_step": "Choose a target column and run AutoML training.",
            "suggested_targets": candidates[:5],
            "unsupervised_available": len(numeric_cols) >= 2 or len(categorical_cols) >= 2,
        }

    if len(numeric_cols) >= 2 or len(categorical_cols) >= 2:
        return {
            "mode": "unsupervised",
            "confidence": "medium",
            "summary": "No obvious target/label column was found, so unsupervised learning is the safer starting point.",
            "recommended_next_step": "Use clustering, anomaly detection, PCA, or association rules to discover structure.",
            "suggested_targets": [],
            "unsupervised_available": True,
        }

    return {
        "mode": "eda_only",
        "confidence": "low",
        "summary": "There is not enough column variety to recommend supervised or unsupervised modelling yet.",
        "recommended_next_step": "Clean or enrich the dataset first, then review the recommendation again.",
        "suggested_targets": [],
        "unsupervised_available": False,
    }


def profile_dataframe(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    total_missing = int(df.isna().sum().sum())
    sample_df = df.sample(PROFILE_SAMPLE_ROWS, random_state=42) if n_rows > PROFILE_SAMPLE_ROWS else df

    columns = [column_profile(df, c, sample_df) for c in df.columns]

    return {
        "shape": {"rows": n_rows, "columns": n_cols},
        "duplicate_rows": duplicate_rows,
        "total_missing_cells": total_missing,
        "total_cells": n_rows * n_cols,
        "columns": columns,
        "describe": describe_dataframe(df),
        "correlation": correlation_matrix(df),
        "preview": safe_preview(df, 15),
        "learning_recommendation": learning_recommendation(df),
    }