"""
Principal Component Analysis utility.

Answers three practical questions for the user:
  1. Is PCA useful here? (need 3+ numeric columns with meaningful variance)
  2. How many components explain 80%/90%/95% of variance?
  3. What does a 2D projection look like?

Fully offline — scikit-learn's PCA, no extra dependencies.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def _prepare(df: pd.DataFrame) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Scale numeric columns, impute missing values, return (matrix, col_names)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Drop constant columns — PCA breaks on them and they add nothing
    num_cols = [c for c in num_cols if df[c].dropna().std() > 0]
    if len(num_cols) < 2:
        return None, None
    X = df[num_cols].copy()
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=num_cols)
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=num_cols)
    return X.values, num_cols


def analyse(df: pd.DataFrame, color_col: str | None = None) -> dict:
    """
    Run PCA on all numeric columns and return a full analysis dict:
      - feasible (bool) + reason if not
      - explained variance per component
      - cumulative variance
      - components needed for 80/90/95%
      - 2D scatter projection (first 2 PCs)
      - loadings for each column (how much it contributes to each PC)
      - verdict + recommendation
    `color_col` (optional categorical/label column) colours the 2D scatter by category.
    """
    X, col_names = _prepare(df)

    if X is None:
        return {
            "feasible": False,
            "reason": "Need at least 2 numeric columns with non-constant values to run PCA.",
            "n_numeric": 0,
        }

    n_samples, n_features = X.shape
    if n_samples < 10:
        return {
            "feasible": False,
            "reason": f"Only {n_samples} rows — PCA needs more data to be meaningful.",
            "n_numeric": n_features,
        }

    n_components = min(n_features, n_samples)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    explained = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    cumulative = [round(float(v) * 100, 2) for v in np.cumsum(pca.explained_variance_ratio_)]

    def components_for(threshold_pct):
        for i, c in enumerate(cumulative):
            if c >= threshold_pct:
                return i + 1
        return n_components

    n_for_80 = components_for(80)
    n_for_90 = components_for(90)
    n_for_95 = components_for(95)

    top2_variance = cumulative[1] if len(cumulative) > 1 else cumulative[0]

    # Verdict
    if n_for_80 <= 2:
        verdict = "excellent"
        recommendation = (
            f"PCA is highly effective here — just 2 components capture {top2_variance:.1f}% of the variance. "
            "You can safely reduce your features to 2 dimensions for visualization or to simplify training."
        )
    elif n_for_80 <= max(3, n_features // 2):
        verdict = "good"
        recommendation = (
            f"{n_for_80} components capture 80% of the variance (down from {n_features} original columns). "
            "PCA is worth using for dimensionality reduction before training."
        )
    elif n_for_90 >= n_features:
        verdict = "poor"
        recommendation = (
            f"PCA is not very helpful here — you need all {n_features} components to explain 90% of the variance. "
            "Your features are either already few, or carry independent information that can't be compressed."
        )
    else:
        verdict = "moderate"
        recommendation = (
            f"{n_for_90} components explain 90% of the variance (down from {n_features} columns). "
            "PCA offers modest compression — useful if training speed matters, but not essential."
        )

    # 2D projection
    pca2 = PCA(n_components=2, random_state=42)
    coords = pca2.fit_transform(X)

    # Group by color column if given
    scatter_points = []
    labels_used = []
    if color_col and color_col in df.columns:
        groups = df[color_col].fillna("(missing)").astype(str)
        labels_used = sorted(groups.unique().tolist())
        for i in range(len(coords)):
            scatter_points.append({
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "label": groups.iloc[i],
            })
    else:
        for i in range(len(coords)):
            scatter_points.append({
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "label": None,
            })

    # Loadings: each column's contribution to the first 2 PCs
    loadings = []
    for j, col in enumerate(col_names):
        loadings.append({
            "column": col,
            "pc1": round(float(pca2.components_[0, j]), 4),
            "pc2": round(float(pca2.components_[1, j]), 4),
        })

    pc1_var = round(float(pca2.explained_variance_ratio_[0]) * 100, 2)
    pc2_var = round(float(pca2.explained_variance_ratio_[1]) * 100, 2) if n_features > 1 else 0.0

    return {
        "feasible": True,
        "n_numeric": n_features,
        "n_samples": n_samples,
        "n_components_total": n_components,
        "explained_variance": explained,
        "cumulative_variance": cumulative,
        "components_for_80": n_for_80,
        "components_for_90": n_for_90,
        "components_for_95": n_for_95,
        "top2_variance": top2_variance,
        "verdict": verdict,                   # "excellent" | "good" | "moderate" | "poor"
        "recommendation": recommendation,
        "scatter": scatter_points,
        "scatter_labels": labels_used,
        "pc1_variance": pc1_var,
        "pc2_variance": pc2_var,
        "loadings": loadings,
        "columns": col_names,
    }