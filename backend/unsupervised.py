"""
Unsupervised learning module: clustering, anomaly detection,
dimensionality reduction, and association rules.

All functions are self-contained and return JSON-serializable results.
"""
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -- Large-dataset limits ------------------------------------------------------
# Some methods need memory / time that grows with the SQUARE of the row count
# (hierarchical, t-SNE, DBSCAN, LOF, silhouette, one-hot association rules).
# Past these sizes they are run on a random sample and every row is then labelled
# from that sample, so a 1,000,000-row file finishes in seconds instead of hanging.
# Every result carries a "sampling" note so nothing is silently approximated.
MAX_PLOT_POINTS = 5000          # points sent to the browser chart (JSON + canvas cost)
MAX_HIERARCHICAL_FIT = 5000     # ward linkage builds an n x n distance matrix
MAX_DBSCAN_FIT = 20000
MAX_TSNE_ROWS = 4000            # t-SNE is ~O(n log n) but with a huge constant
MAX_LOF_FIT = 8000
MAX_K_SEARCH_ROWS = 20000       # trying K = 2..10 on the full data is wasted work
MAX_AUTO_SEARCH_ROWS = 5000     # "Auto best" tries ~25 candidates
MAX_ASSOC_ROWS = 50000
MAX_ASSOC_CARDINALITY = 20      # a column with 5,000 distinct values is an ID, not a category
MAX_ASSOC_ITEMS = 60            # one-hot columns fed to Apriori


# ── Helpers ────────────────────────────────────────────────────────────────

def _numeric_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return scaled numeric matrix and column names."""
    generated_cols = {"cluster_label"}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in generated_cols]
    if len(num_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for unsupervised analysis.")
    X = df[num_cols].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, num_cols


def preprocessing_summary(df: pd.DataFrame) -> dict:
    """Describe preprocessing used by numeric unsupervised algorithms."""
    generated_cols = {"cluster_label"}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in generated_cols]
    return {
        "scaling": "StandardScaler",
        "scaling_description": "Numeric features are median-imputed, then standardized to mean 0 and standard deviation 1 before distance-based unsupervised methods run.",
        "features_used": num_cols,
        "n_numeric_features": len(num_cols),
        "excluded_generated_columns": sorted(generated_cols & set(df.columns)),
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
    if pd.isna(val):
        return None
    return val


def _cluster_sizes(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    return {
        ("Outliers" if int(label) == -1 else f"Cluster {int(label)}"): int(count)
        for label, count in zip(unique, counts)
    }


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_indices(n: int, limit: int, seed: int = 42) -> np.ndarray:
    """Sorted random row positions (all rows when n <= limit)."""
    if n <= limit:
        return np.arange(n)
    return np.sort(_rng(seed).choice(n, size=limit, replace=False))


def _sampling_info(n_rows: int, fit_rows: int, plotted: int | None = None, why: str = "") -> dict:
    """What was approximated, in words a non-technical reader can follow."""
    info = {
        "n_rows": int(n_rows),
        "fit_rows": int(fit_rows),
        "plotted_points": int(plotted) if plotted is not None else None,
        "sampled": bool(fit_rows < n_rows),
        "note": "",
    }
    parts = []
    if fit_rows < n_rows:
        parts.append(
            f"Model fitted on a random sample of {fit_rows:,} of {n_rows:,} rows"
            + (f" ({why})" if why else "")
            + "; every row was then labelled from that sample."
        )
    if plotted is not None and plotted < n_rows:
        parts.append(f"Chart shows {plotted:,} representative points, not all {n_rows:,}.")
    info["note"] = " ".join(parts)
    return info


def _plot_indices(labels: np.ndarray | None, n: int, limit: int = MAX_PLOT_POINTS, always: np.ndarray | None = None) -> np.ndarray:
    """Row positions to draw. Small groups (and anything in `always`, e.g. anomalies)
    keep their points so they never vanish from a chart of a big dataset."""
    if n <= limit:
        return np.arange(n)
    rng = _rng(7)
    keep = np.zeros(n, dtype=bool)
    if always is not None and len(always):
        always = np.asarray(always)
        if len(always) > limit // 2:
            always = rng.choice(always, size=limit // 2, replace=False)
        keep[always] = True
    if labels is None:
        labels = np.zeros(n, dtype=int)
    for lab in np.unique(labels):
        members = np.where((labels == lab) & ~keep)[0]
        quota = max(100, int(limit * len(members) / n))
        if len(members) > quota:
            members = rng.choice(members, size=quota, replace=False)
        keep[members] = True
    idx = np.where(keep)[0]
    if len(idx) > limit:
        forced = np.where(keep)[0] if always is None else np.intersect1d(idx, always)
        rest = np.setdiff1d(idx, forced)
        idx = np.sort(np.concatenate([forced, rng.choice(rest, size=max(0, limit - len(forced)), replace=False)]))
    return idx


def _scatter_points(X: np.ndarray, idx: np.ndarray, labels: np.ndarray | None) -> list[dict]:
    return [
        {
            "x": round(float(X[i, 0]), 4),
            "y": round(float(X[i, 1]), 4),
            "cluster": int(labels[i]) if labels is not None else 0,
        }
        for i in idx
    ]


def _extend_by_neighbour(X_fit: np.ndarray, labels_fit: np.ndarray, X_all: np.ndarray, max_distance: float | None = None) -> np.ndarray:
    """Give every row the label of its nearest fitted row (1-NN). With
    `max_distance`, rows farther than that from any fitted row become noise (-1)."""
    knn = KNeighborsClassifier(n_neighbors=1).fit(X_fit, labels_fit)
    out = knn.predict(X_all)
    if max_distance is not None:
        dist, _ = knn.kneighbors(X_all, n_neighbors=1)
        out = np.where(dist[:, 0] <= max_distance, out, -1)
    return out.astype(int)


def _clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Return standard clustering metrics when labels are valid for scoring."""
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return {}
    return {
        "silhouette": round(float(silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
    }


# ── Clustering ─────────────────────────────────────────────────────────────

def cluster_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    """K-means clustering. Scales to millions of rows (cost grows linearly)."""
    X, cols = _numeric_matrix(df)
    n = len(X)
    n_clusters = max(2, min(int(n_clusters), n - 1))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10 if n <= 50000 else 3)
    labels = model.fit_predict(X)

    plot_idx = _plot_indices(labels, n)
    result = {
        "method": "K-Means",
        "preprocessing": preprocessing_summary(df),
        "n_clusters": n_clusters,
        "labels": [int(l) for l in labels],
        "cluster_sizes": _cluster_sizes(labels),
        "metrics": _clustering_metrics(X, labels),
        "centers": model.cluster_centers_.tolist(),
        "inertia": round(float(model.inertia_), 4),
        "sampling": _sampling_info(n, n, len(plot_idx)),
        "visualization": {
            "type": "scatter",
            "title": f"K-Means Clustering ({n_clusters} clusters)",
            "points": _scatter_points(X, plot_idx, labels),
            "x_label": cols[0] if len(cols) > 0 else "Feature 1",
            "y_label": cols[1] if len(cols) > 1 else "Feature 2",
        },
    }
    return result


def _dbscan_fit_extend(X: np.ndarray, eps: float, min_samples: int, fit_limit: int = MAX_DBSCAN_FIT):
    """DBSCAN on a sample when the data is big, then label every row.

    min_samples is scaled by the sampling fraction (a 10% sample has ~10% as many
    neighbours per point); eps is left alone because it is a distance. Rows outside
    the sample take the label of a core point within eps, otherwise stay noise --
    the same rule DBSCAN uses for border points.
    Returns (labels for all rows, min_samples actually used, core_X, core_labels).
    """
    n = len(X)
    idx = _sample_indices(n, fit_limit)
    frac = len(idx) / n
    ms = int(min_samples) if frac >= 1 else max(3, int(round(min_samples * frac)))
    model = DBSCAN(eps=eps, min_samples=ms).fit(X[idx])
    fit_labels = model.labels_
    core_X = X[idx][model.core_sample_indices_]
    core_labels = fit_labels[model.core_sample_indices_]
    if frac >= 1:
        return fit_labels.astype(int), ms, core_X, core_labels, len(idx)
    if len(core_X) == 0:
        return np.full(n, -1, dtype=int), ms, core_X, core_labels, len(idx)
    labels = _extend_by_neighbour(core_X, core_labels, X, max_distance=eps)
    labels[idx] = fit_labels                      # sampled rows keep their exact DBSCAN label
    return labels, ms, core_X, core_labels, len(idx)


def cluster_dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> dict:
    """DBSCAN clustering - automatically detects number of clusters + outliers."""
    X, cols = _numeric_matrix(df)
    n = len(X)
    labels, ms_used, _, _, fit_rows = _dbscan_fit_extend(X, float(eps), int(min_samples))

    plot_idx = _plot_indices(labels, n, always=np.where(labels == -1)[0][: MAX_PLOT_POINTS // 4])
    result = {
        "method": "DBSCAN",
        "preprocessing": preprocessing_summary(df),
        "eps": eps,
        "min_samples": min_samples,
        "labels": [int(l) for l in labels],
        "cluster_sizes": _cluster_sizes(labels),
        "n_outliers": int(np.sum(labels == -1)),
        "metrics": {},
        "sampling": _sampling_info(
            n, fit_rows, len(plot_idx),
            why=f"DBSCAN compares every point with its neighbours; min_samples was scaled from {min_samples} to {ms_used} to match" if fit_rows < n else "",
        ),
    }
    non_noise = labels != -1
    if np.sum(non_noise) >= 2 and len(np.unique(labels[non_noise])) > 1:
        result["metrics"] = _clustering_metrics(X[non_noise], labels[non_noise])

    result["visualization"] = {
        "type": "scatter",
        "title": f"DBSCAN Clustering (eps={eps})",
        "points": _scatter_points(X, plot_idx, labels),
        "x_label": cols[0] if len(cols) > 0 else "Feature 1",
        "y_label": cols[1] if len(cols) > 1 else "Feature 2",
    }
    return result


def cluster_hierarchical(df: pd.DataFrame, n_clusters: int = 3, linkage: str = "ward") -> dict:
    """Hierarchical (agglomerative) clustering.

    The method builds an n x n distance table, so 60,000 rows would need ~13 GB.
    Above MAX_HIERARCHICAL_FIT rows it is fitted on a random sample and the rest
    of the rows join the cluster of their nearest sampled row.
    """
    X, cols = _numeric_matrix(df)
    n = len(X)
    n_clusters = max(2, min(int(n_clusters), n - 1))
    idx = _sample_indices(n, MAX_HIERARCHICAL_FIT)
    fit_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(X[idx])
    if len(idx) < n:
        labels = _extend_by_neighbour(X[idx], fit_labels, X)
        labels[idx] = fit_labels
    else:
        labels = fit_labels

    plot_idx = _plot_indices(labels, n)
    return {
        "method": "Hierarchical",
        "preprocessing": preprocessing_summary(df),
        "linkage": linkage,
        "n_clusters": n_clusters,
        "labels": [int(l) for l in labels],
        "cluster_sizes": _cluster_sizes(labels),
        "metrics": _clustering_metrics(X, labels),
        "sampling": _sampling_info(n, len(idx), len(plot_idx), why="hierarchical clustering needs memory that grows with rows squared"),
        "visualization": {
            "type": "scatter",
            "title": f"Hierarchical Clustering ({n_clusters} clusters)",
            "points": _scatter_points(X, plot_idx, labels),
            "x_label": cols[0] if len(cols) > 0 else "Feature 1",
            "y_label": cols[1] if len(cols) > 1 else "Feature 2",
        },
    }


def suggest_clusters(df: pd.DataFrame, max_clusters: int = 10) -> dict:
    """Analyze and suggest the optimal number of clusters using multiple methods.
    
    Returns:
    - Elbow method data (inertia vs K)
    - Silhouette scores for each K
    - Davies-Bouldin scores for each K
    - Recommendation for best K
    """
    X, cols = _numeric_matrix(df)
    if len(X) < 3:
        raise ValueError("Need at least 3 rows for cluster analysis.")
    
    n_total = len(X)
    upper_k = max(2, min(int(max_clusters), len(X) - 1))

    # Trying K = 2..10 on every row of a big file is wasted work: the shape of the
    # elbow / silhouette curve is the same on a 20,000-row sample.
    scan_idx = _sample_indices(n_total, MAX_K_SEARCH_ROWS)
    X = X[scan_idx]
    scan_n_init = 10 if len(X) == n_total else 3

    inertias = []
    silhouettes = []
    davies_bouldins = []
    calinski_harabasz = []
    k_values = list(range(2, upper_k + 1))
    
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=scan_n_init)
        labels = model.fit_predict(X)
        inertias.append(round(float(model.inertia_), 2))
        
        metrics = _clustering_metrics(X, labels)
        if metrics:
            silhouettes.append(round(metrics.get("silhouette", 0), 4))
            davies_bouldins.append(round(metrics.get("davies_bouldin", 0), 4))
            calinski_harabasz.append(round(metrics.get("calinski_harabasz", 0), 4))
        else:
            silhouettes.append(0)
            davies_bouldins.append(0)
            calinski_harabasz.append(0)
    
    # Find best K using multiple criteria
    # 1. Silhouette score (higher is better)
    best_silhouette_idx = silhouettes.index(max(silhouettes)) if max(silhouettes) > 0 else 0
    best_silhouette_k = k_values[best_silhouette_idx]
    
    # 2. Davies-Bouldin index (lower is better)
    valid_db = [(i, v) for i, v in enumerate(davies_bouldins) if v > 0]
    if valid_db:
        best_db_idx, best_db_value = min(valid_db, key=lambda x: x[1])
        best_db_k = k_values[best_db_idx]
    else:
        best_db_k = k_values[0]
    
    # 3. Elbow method (look for the point where inertia decrease slows down)
    # Calculate second derivative to find the "elbow"
    if len(inertias) >= 3:
        diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        second_diffs = [diffs[i] - diffs[i+1] for i in range(len(diffs)-1)]
        if second_diffs:
            elbow_idx = second_diffs.index(max(second_diffs)) + 1
            elbow_k = k_values[elbow_idx]
        else:
            elbow_k = k_values[0]
    else:
        elbow_k = k_values[0]
    
    # 4. Calinski-Harabasz score (higher is better)
    best_ch_idx = calinski_harabasz.index(max(calinski_harabasz)) if max(calinski_harabasz) > 0 else 0
    best_ch_k = k_values[best_ch_idx]

    # Consensus: choose the K that appears most frequently across methods.
    # Ties are broken by silhouette because it is the easiest metric to compare
    # across K values for compact, separated clusters.
    k_votes = [best_silhouette_k, best_db_k, elbow_k, best_ch_k]
    from collections import Counter
    vote_counts = Counter(k_votes)
    top_votes = vote_counts.most_common()
    max_votes = top_votes[0][1]
    tied_k = {k for k, votes in top_votes if votes == max_votes}
    if len(tied_k) > 1:
        best_k = max(tied_k, key=lambda k: silhouettes[k_values.index(k)])
    else:
        best_k = top_votes[0][0]
    
    # Generate recommendation reason
    reasons = []
    if best_k == best_silhouette_k:
        reasons.append(f"highest silhouette score ({silhouettes[best_silhouette_idx]})")
    if best_k == best_db_k:
        reasons.append(f"lowest Davies-Bouldin index ({best_db_value})")
    if best_k == elbow_k:
        reasons.append("elbow point in inertia curve")
    if best_k == best_ch_k:
        reasons.append(f"highest Calinski-Harabasz score ({calinski_harabasz[best_ch_idx]})")
    
    recommendation = f"K={best_k} is suggested based on: {', '.join(reasons) or 'the best consensus across cluster-quality metrics'}"
    k_analysis = [
        {
            "k": int(k),
            "inertia": inertias[i],
            "silhouette": silhouettes[i],
            "davies_bouldin": davies_bouldins[i],
            "calinski_harabasz": calinski_harabasz[i],
            "votes": int(vote_counts.get(k, 0)),
        }
        for i, k in enumerate(k_values)
    ]
    
    return {
        "method": "Cluster Number Analysis",
        "preprocessing": preprocessing_summary(df),
        "sampling": _sampling_info(n_total, len(X), why="trying every K on all rows adds time but not information"),
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "davies_bouldins": davies_bouldins,
        "calinski_harabasz": calinski_harabasz,
        "k_analysis": k_analysis,
        "recommended_k": best_k,
        "best_silhouette_k": best_silhouette_k,
        "best_silhouette_score": max(silhouettes) if silhouettes else 0,
        "best_db_k": best_db_k,
        "best_db_score": min(valid_db, key=lambda x: x[1])[1] if valid_db else 0,
        "best_ch_k": best_ch_k,
        "best_ch_score": max(calinski_harabasz) if calinski_harabasz else 0,
        "elbow_k": elbow_k,
        "recommendation": recommendation,
        "visualization": {
            "type": "elbow_curve",
            "title": f"Cluster Number Analysis (Recommended K={best_k})",
            "k_values": k_values,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "davies_bouldins": davies_bouldins,
            "recommended_k": best_k,
        }
    }


def cluster_best(df: pd.DataFrame, max_clusters: int = 10) -> dict:
    """Try sensible clustering options and return the best scored result.

    Silhouette is the primary selection metric because it is comparable across
    K-means and hierarchical clustering. Davies-Bouldin breaks close ties.

    The search itself (about 25 candidates) runs on at most MAX_AUTO_SEARCH_ROWS
    rows. The winning method is then applied to ALL rows, so the labels cover the
    whole dataset while the button stays responsive on files of any size.
    """
    from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage

    X_all, cols = _numeric_matrix(df)
    n = len(X_all)
    if n < 3:
        raise ValueError("Need at least 3 rows for automatic clustering.")

    search_idx = _sample_indices(n, MAX_AUTO_SEARCH_ROWS)
    X = X_all[search_idx]
    sampled = len(search_idx) < n
    upper_k = max(2, min(int(max_clusters), len(X) - 1))
    candidates = []

    for k in range(2, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = model.fit_predict(X)
        metrics = _clustering_metrics(X, labels)
        if metrics:
            candidates.append({
                "method": "K-Means", "n_clusters": k, "labels": labels,
                "metrics": metrics, "inertia": round(float(model.inertia_), 4),
            })

    # One ward tree, cut at every K -- instead of re-fitting the whole tree 9 times.
    tree = scipy_linkage(X, method="ward")
    for k in range(2, upper_k + 1):
        labels = fcluster(tree, t=k, criterion="maxclust") - 1
        metrics = _clustering_metrics(X, labels)
        if metrics:
            candidates.append({
                "method": "Hierarchical", "n_clusters": k, "labels": labels,
                "metrics": metrics, "linkage": "ward",
            })

    base_min_samples = max(5, min(20, len(cols) * 2))
    frac = len(X) / n
    min_samples = base_min_samples if not sampled else max(3, int(round(base_min_samples * frac)))
    for eps in (0.3, 0.5, 0.8, 1.2, 1.8, 2.5):
        model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = model.labels_
        non_noise = labels != -1
        n_clusters = len(set(labels[non_noise]))
        if n_clusters < 2 or np.sum(non_noise) < max(3, int(0.4 * len(labels))):
            continue
        metrics = _clustering_metrics(X[non_noise], labels[non_noise])
        if metrics:
            candidates.append({
                "method": "DBSCAN", "eps": eps, "min_samples": min_samples,
                "n_clusters": n_clusters, "labels": labels, "metrics": metrics,
                "n_outliers": int(np.sum(labels == -1)),
                "_core_X": X[model.core_sample_indices_],
                "_core_labels": labels[model.core_sample_indices_],
            })

    if not candidates:
        raise ValueError("Could not find a valid clustering structure. Try K-Means manually with a chosen K.")

    def rank(candidate):
        metrics = candidate["metrics"]
        return (
            metrics.get("silhouette", -1),
            -metrics.get("davies_bouldin", float("inf")),
            metrics.get("calinski_harabasz", -1),
        )

    candidates.sort(key=rank, reverse=True)
    best = candidates[0]
    search_labels = best.pop("labels")
    core_X = best.pop("_core_X", None)
    core_labels = best.pop("_core_labels", None)

    # Apply the winner to every row.
    if not sampled:
        labels, final_metrics = search_labels, best["metrics"]
    else:
        if best["method"] == "K-Means":
            labels = KMeans(n_clusters=best["n_clusters"], random_state=42, n_init=3).fit_predict(X_all)
        elif best["method"] == "DBSCAN":
            if core_X is not None and len(core_X):
                labels = _extend_by_neighbour(core_X, core_labels, X_all, max_distance=best["eps"])
            else:
                labels = np.full(n, -1, dtype=int)
            labels[search_idx] = search_labels
        else:  # Hierarchical
            labels = _extend_by_neighbour(X, search_labels, X_all)
            labels[search_idx] = search_labels
        keep = labels != -1 if best["method"] == "DBSCAN" else np.ones(n, dtype=bool)
        final_metrics = _clustering_metrics(X_all[keep], labels[keep]) or best["metrics"]

    plot_idx = _plot_indices(labels, n)
    reason = (
        f"Selected {best['method']} because it had the strongest silhouette score "
        f"among {len(candidates)} valid clustering candidates."
    )
    if sampled:
        reason += f" The candidates were compared on a random sample of {len(X):,} rows and the winner was then applied to all {n:,} rows."

    result = {
        "method": "Auto Best Clustering",
        "preprocessing": preprocessing_summary(df),
        "selected_method": best["method"],
        "n_clusters": int(best.get("n_clusters", len(np.unique(labels)))),
        "labels": [int(l) for l in labels],
        "cluster_sizes": _cluster_sizes(labels),
        "metrics": final_metrics,
        "features_used": cols,
        "selection_reason": reason,
        "sampling": _sampling_info(n, len(X), len(plot_idx), why="the search compares ~25 candidates"),
        "candidates": [
            {
                "method": c["method"],
                "n_clusters": int(c.get("n_clusters", 0)),
                "silhouette": c["metrics"].get("silhouette"),
                "davies_bouldin": c["metrics"].get("davies_bouldin"),
                "calinski_harabasz": c["metrics"].get("calinski_harabasz"),
            }
            for c in candidates[:8]
        ],
        "visualization": {
            "type": "scatter",
            "title": f"Auto Best: {best['method']} ({int(best.get('n_clusters', 0))} clusters)",
            "points": _scatter_points(X_all, plot_idx, labels),
            "x_label": cols[0] if len(cols) > 0 else "Feature 1",
            "y_label": cols[1] if len(cols) > 1 else "Feature 2",
        },
    }
    if best["method"] == "DBSCAN":
        result["n_outliers"] = int(np.sum(labels == -1))
    for key in ("eps", "min_samples", "inertia", "linkage"):
        if key in best:
            result[key] = best[key]
    return result


# ── Anomaly Detection ──────────────────────────────────────────────────────

MAX_RETURNED_ANOMALY_ROWS = 20000     # row numbers / scores sent back; the true total is reported separately


def _anomaly_result(method: str, title_name: str, df, X, cols, labels, scores, extra: dict, sampling: dict) -> dict:
    anomaly_pos = np.where(labels == -1)[0]
    n_outliers = int(len(anomaly_pos))
    plot_idx = _plot_indices(np.where(labels == -1, -1, 1), len(X), always=anomaly_pos)
    result = {
        "method": method,
        "preprocessing": preprocessing_summary(df),
        **extra,
        "n_outliers": n_outliers,
        "n_normal": int(np.sum(labels == 1)),
        "outlier_percentage": round(n_outliers / len(df) * 100, 2),
        "anomaly_indices": [int(i) for i in anomaly_pos[:MAX_RETURNED_ANOMALY_ROWS]],
        "anomaly_indices_total": n_outliers,
        "sampling": sampling,
        "visualization": {
            "type": "scatter",
            "title": f"{title_name} Anomaly Detection ({n_outliers} anomalies)",
            "points": _scatter_points(X, plot_idx, np.where(labels == -1, -1, 1)),   # -1 = anomaly, 1 = normal
            "x_label": cols[0] if len(cols) > 0 else "Feature 1",
            "y_label": cols[1] if len(cols) > 1 else "Feature 2",
        },
    }
    if scores is not None:
        result["anomaly_scores"] = [round(float(v), 4) for v in scores[anomaly_pos[:MAX_RETURNED_ANOMALY_ROWS]]]
    return result


def detect_anomalies_isolation_forest(df: pd.DataFrame, contamination: float = 0.1) -> dict:
    """Isolation Forest anomaly detection. Each tree only ever looks at 256 rows,
    so training cost does not grow with the data; scoring is linear."""
    X, cols = _numeric_matrix(df)
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = model.fit_predict(X)
    scores = model.decision_function(X)
    plot_idx_count = min(len(X), MAX_PLOT_POINTS)
    sampling = _sampling_info(len(X), len(X), plot_idx_count if len(X) > MAX_PLOT_POINTS else None)
    return _anomaly_result("Isolation Forest", "Isolation Forest", df, X, cols, labels, scores,
                           {"contamination": contamination}, sampling)


def detect_anomalies_lof(df: pd.DataFrame, contamination: float = 0.1, n_neighbors: int = 20) -> dict:
    """Local Outlier Factor anomaly detection.

    LOF compares every row with its neighbours, which gets very slow on big data.
    Above MAX_LOF_FIT rows the model learns "what normal density looks like" from a
    random sample (novelty mode) and then scores every row against it.
    """
    X, cols = _numeric_matrix(df)
    n = len(X)
    idx = _sample_indices(n, MAX_LOF_FIT)
    k = max(1, min(int(n_neighbors), len(idx) - 1))
    if len(idx) == n:
        model = LocalOutlierFactor(n_neighbors=k, contamination=contamination, n_jobs=-1)
        labels = model.fit_predict(X)
        scores = None
    else:
        model = LocalOutlierFactor(n_neighbors=k, contamination=contamination, novelty=True, n_jobs=-1)
        model.fit(X[idx])
        labels = model.predict(X)
        scores = model.decision_function(X)
    sampling = _sampling_info(n, len(idx), min(n, MAX_PLOT_POINTS) if n > MAX_PLOT_POINTS else None,
                              why="LOF compares every row with its neighbours, which is slow on big data")
    return _anomaly_result("Local Outlier Factor", "Local Outlier Factor", df, X, cols, labels, scores,
                           {"n_neighbors": n_neighbors, "contamination": contamination}, sampling)


# ── Dimensionality Reduction ───────────────────────────────────────────────

def reduce_tsne(df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0) -> dict:
    """t-SNE dimensionality reduction for visualization.

    t-SNE is a picture tool, not a scalable model: 30,000 rows already take minutes.
    It runs on a random sample of up to MAX_TSNE_ROWS rows (after a PCA step to 30
    dimensions when there are many columns, the standard speed-up).
    """
    X, cols = _numeric_matrix(df)
    n = len(X)
    idx = _sample_indices(n, MAX_TSNE_ROWS)
    Xs = X[idx]
    if Xs.shape[1] > 30:
        Xs = PCA(n_components=30, random_state=42).fit_transform(Xs)
    perplexity = float(min(perplexity, max(2, len(Xs) - 1)))
    model = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_jobs=-1)
    embedding = model.fit_transform(Xs)

    points = [
        {"x": round(float(embedding[i, 0]), 4), "y": round(float(embedding[i, 1]), 4), "index": int(idx[i])}
        for i in range(len(embedding))
    ]
    return {
        "method": "t-SNE",
        "preprocessing": preprocessing_summary(df),
        "n_components": n_components,
        "perplexity": perplexity,
        "points": points,
        "sampling": _sampling_info(n, len(idx), why="t-SNE is a visualisation tool that becomes very slow beyond a few thousand rows"),
        "visualization": {
            "type": "scatter",
            "title": f"t-SNE Projection (perplexity={perplexity:g})",
            "points": [{"x": p["x"], "y": p["y"], "cluster": 0} for p in points],
            "x_label": "t-SNE 1",
            "y_label": "t-SNE 2",
        },
    }


MAX_PCA_FIT_ROWS = 200000


def reduce_pca_advanced(df: pd.DataFrame, n_components: int = 2) -> dict:
    """Advanced PCA with variance explained. Fitted on up to 200k rows (more adds
    nothing to a covariance estimate); only a sample of points is drawn."""
    X, cols = _numeric_matrix(df)
    n = len(X)
    n_components = max(2, min(int(n_components), X.shape[1], n))
    fit_idx = _sample_indices(n, MAX_PCA_FIT_ROWS)
    pca = PCA(n_components=n_components, random_state=42).fit(X[fit_idx])
    plot_idx = _plot_indices(None, n)
    embedding = pca.transform(X[plot_idx])

    explained_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    points = [
        {"x": round(float(embedding[i, 0]), 4), "y": round(float(embedding[i, 1]), 4), "index": int(plot_idx[i])}
        for i in range(len(embedding))
    ]
    return {
        "method": "PCA",
        "preprocessing": preprocessing_summary(df),
        "n_components": n_components,
        "explained_variance": explained_var,
        "cumulative_variance": [round(float(v), 2) for v in np.cumsum(pca.explained_variance_ratio_)],
        "points": points,
        "sampling": _sampling_info(n, len(fit_idx), len(plot_idx)),
        "visualization": {
            "type": "scatter",
            "title": "PCA Projection (PC1 vs PC2)",
            "points": [{"x": p["x"], "y": p["y"], "cluster": 0} for p in points],
            "x_label": f"PC1 ({explained_var[0]}%)",
            "y_label": f"PC2 ({explained_var[1]}%)" if len(explained_var) > 1 else "PC2",
        },
    }


# ── Association Rules ──────────────────────────────────────────────────────

def association_rules(df: pd.DataFrame, min_support: float = 0.1, min_confidence: float = 0.5) -> dict:
    """Frequent itemset mining for categorical/boolean columns (Apriori).

    Apriori works on one-hot "item" columns. Two things make that explode on real
    files, so both are bounded: (1) a column with thousands of distinct values (an
    ID, a name, a date) would create thousands of item columns and an n x n table,
    so only columns with <= MAX_ASSOC_CARDINALITY values are used; (2) rows are
    sampled above MAX_ASSOC_ROWS, which barely changes support percentages.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules as ar_rules
    except ImportError:
        raise ImportError("mlxtend is required for association rules. Install: pip install mlxtend")

    n_total = len(df)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        raise ValueError("Need at least one categorical/boolean column for association rules.")

    sample = df[cat_cols].iloc[_sample_indices(n_total, MAX_ASSOC_ROWS)]
    usable, skipped = [], []
    for c in cat_cols:
        k = int(sample[c].nunique(dropna=True))
        (usable if 1 <= k <= MAX_ASSOC_CARDINALITY else skipped).append(c)
    if not usable:
        raise ValueError(
            "No suitable categorical columns: every one has more than "
            f"{MAX_ASSOC_CARDINALITY} distinct values (looks like IDs or free text): {', '.join(map(str, skipped[:8]))}."
        )

    encoded = pd.get_dummies(sample[usable], prefix_sep="=", dummy_na=False, dtype=bool)
    # Keep items that vary, and that appear often enough to ever reach min_support.
    support = encoded.mean()
    keep = [c for c in encoded.columns if 0 < support[c] < 1 and support[c] >= min_support]
    if not keep:
        raise ValueError("Could not create binary columns from categorical data (try a lower min support).")
    keep = sorted(keep, key=lambda c: support[c], reverse=True)[:MAX_ASSOC_ITEMS]
    df_binary = encoded[keep]

    notes = []
    if skipped:
        notes.append(f"Skipped {len(skipped)} column(s) with more than {MAX_ASSOC_CARDINALITY} distinct values (IDs / free text): {', '.join(map(str, skipped[:8]))}{'...' if len(skipped) > 8 else ''}.")
    if len(sample) < n_total:
        notes.append(f"Rules were mined from a random sample of {len(sample):,} of {n_total:,} rows.")
    if len(encoded.columns) > len(keep) and len(keep) == MAX_ASSOC_ITEMS:
        notes.append(f"Only the {MAX_ASSOC_ITEMS} most frequent items were used.")
    base = {"method": "Apriori", "min_support": min_support, "min_confidence": min_confidence,
            "skipped_columns": skipped, "sampling": _sampling_info(n_total, len(sample)), "notes": notes}
    if notes and not base["sampling"]["note"]:
        base["sampling"]["note"] = " ".join(notes)

    frequent_itemsets = apriori(df_binary, min_support=min_support, use_colnames=True, max_len=3)
    if frequent_itemsets.empty:
        return {**base, "n_rules": 0, "rules": [],
                "message": f"No frequent itemsets found with min_support={min_support}. Try lowering it."}

    rules = ar_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return {**base, "n_rules": 0, "rules": [],
                "message": f"No rules found with min_confidence={min_confidence}. Try lowering it."}

    formatted_rules = [
        {
            "antecedents": list(row["antecedents"]),
            "consequents": list(row["consequents"]),
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 4),
        }
        for _, row in rules.iterrows()
    ]
    formatted_rules.sort(key=lambda x: x["lift"], reverse=True)
    return {**base, "n_rules": len(formatted_rules), "rules": formatted_rules[:50]}    # top 50 by lift


# ── Auto-select best method ────────────────────────────────────────────────

def suggest_unsupervised(df: pd.DataFrame) -> dict:
    """Suggest which unsupervised methods might be useful for this dataset."""
    suggestions = []
    n_numeric = len(df.select_dtypes(include=[np.number]).columns)
    n_categorical = len(df.select_dtypes(include=["object", "category", "bool"]).columns)
    n_rows = len(df)

    if n_numeric >= 2:
        suggestions.append({
            "task": "Clustering",
            "reason": f"You have {n_numeric} numeric columns - K-means or Hierarchical clustering can group similar rows.",
            "methods": ["K-Means", "Hierarchical", "DBSCAN"],
        })
        suggestions.append({
            "task": "Anomaly Detection",
            "reason": f"With {n_numeric} numeric columns, Isolation Forest can spot unusual rows.",
            "methods": ["Isolation Forest", "Local Outlier Factor"],
        })
        suggestions.append({
            "task": "Dimensionality Reduction",
            "reason": "t-SNE or PCA can project high-dimensional data into 2D for visualization.",
            "methods": ["t-SNE", "PCA"],
        })

    if n_categorical >= 2:
        suggestions.append({
            "task": "Association Rules",
            "reason": f"You have {n_categorical} categorical columns - Apriori can find relationships between categories.",
            "methods": ["Apriori"],
        })

    return {
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "n_rows": n_rows,
        "suggestions": suggestions,
    }
