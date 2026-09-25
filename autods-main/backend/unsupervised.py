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
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


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
    """K-means clustering."""
    X, cols = _numeric_matrix(df)
    n_clusters = max(2, min(int(n_clusters), len(X) - 1))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    result = {
        "method": "K-Means",
        "preprocessing": preprocessing_summary(df),
        "n_clusters": n_clusters,
        "labels": [int(l) for l in labels],
        "cluster_sizes": {},
        "metrics": {},
    }
    result["cluster_sizes"] = _cluster_sizes(labels)
    result["metrics"] = _clustering_metrics(X, labels)

    result["centers"] = model.cluster_centers_.tolist()
    result["inertia"] = round(float(model.inertia_), 4)
    
    # Add visualization data
    result["visualization"] = {
        "type": "scatter",
        "title": f"K-Means Clustering ({n_clusters} clusters)",
        "points": [
            {
                "x": round(float(X[i, 0]), 4),
                "y": round(float(X[i, 1]), 4),
                "cluster": int(labels[i])
            }
            for i in range(len(X))
        ],
        "x_label": cols[0] if len(cols) > 0 else "Feature 1",
        "y_label": cols[1] if len(cols) > 1 else "Feature 2",
    }
    
    return result


def cluster_dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> dict:
    """DBSCAN clustering - automatically detects number of clusters + outliers."""
    X, cols = _numeric_matrix(df)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    result = {
        "method": "DBSCAN",
        "preprocessing": preprocessing_summary(df),
        "eps": eps,
        "min_samples": min_samples,
        "labels": [int(l) for l in labels],
        "cluster_sizes": {},
        "n_outliers": int(np.sum(labels == -1)),
        "metrics": {},
    }
    unique = np.unique(labels)
    for u in unique:
        mask = labels == u
        label_name = "Outliers" if u == -1 else f"Cluster {int(u)}"
        result["cluster_sizes"][label_name] = int(np.sum(mask))

    non_noise = labels != -1
    if np.sum(non_noise) >= 2 and len(np.unique(labels[non_noise])) > 1:
        result["metrics"] = _clustering_metrics(X[non_noise], labels[non_noise])

    # Add visualization data
    result["visualization"] = {
        "type": "scatter",
        "title": f"DBSCAN Clustering (eps={eps})",
        "points": [
            {
                "x": round(float(X[i, 0]), 4),
                "y": round(float(X[i, 1]), 4),
                "cluster": int(labels[i])
            }
            for i in range(len(X))
        ],
        "x_label": cols[0] if len(cols) > 0 else "Feature 1",
        "y_label": cols[1] if len(cols) > 1 else "Feature 2",
    }
    
    return result


def cluster_hierarchical(df: pd.DataFrame, n_clusters: int = 3, linkage: str = "ward") -> dict:
    """Hierarchical (agglomerative) clustering."""
    X, cols = _numeric_matrix(df)
    n_clusters = max(2, min(int(n_clusters), len(X) - 1))
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)

    result = {
        "method": "Hierarchical",
        "preprocessing": preprocessing_summary(df),
        "linkage": linkage,
        "n_clusters": n_clusters,
        "labels": [int(l) for l in labels],
        "cluster_sizes": {},
        "metrics": {},
    }
    result["cluster_sizes"] = _cluster_sizes(labels)
    result["metrics"] = _clustering_metrics(X, labels)
    
    # Add visualization data
    result["visualization"] = {
        "type": "scatter",
        "title": f"Hierarchical Clustering ({n_clusters} clusters)",
        "points": [
            {
                "x": round(float(X[i, 0]), 4),
                "y": round(float(X[i, 1]), 4),
                "cluster": int(labels[i])
            }
            for i in range(len(X))
        ],
        "x_label": cols[0] if len(cols) > 0 else "Feature 1",
        "y_label": cols[1] if len(cols) > 1 else "Feature 2",
    }
    
    return result


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
    
    upper_k = max(2, min(int(max_clusters), len(X) - 1))
    
    inertias = []
    silhouettes = []
    davies_bouldins = []
    calinski_harabasz = []
    k_values = list(range(2, upper_k + 1))
    
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
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
    """
    X, cols = _numeric_matrix(df)
    if len(X) < 3:
        raise ValueError("Need at least 3 rows for automatic clustering.")

    upper_k = max(2, min(int(max_clusters), len(X) - 1))
    candidates = []

    for k in range(2, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        metrics = _clustering_metrics(X, labels)
        if metrics:
            candidates.append({
                "method": "K-Means",
                "n_clusters": k,
                "labels": labels,
                "metrics": metrics,
                "inertia": round(float(model.inertia_), 4),
            })

    # Agglomerative clustering is useful, but it is quadratic-ish; keep it for
    # smaller datasets so the "auto" button stays responsive.
    if len(X) <= 5000:
        for k in range(2, upper_k + 1):
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(X)
            metrics = _clustering_metrics(X, labels)
            if metrics:
                candidates.append({
                    "method": "Hierarchical",
                    "n_clusters": k,
                    "labels": labels,
                    "metrics": metrics,
                    "linkage": "ward",
                })

    min_samples = max(5, min(20, len(cols) * 2))
    for eps in (0.3, 0.5, 0.8, 1.2, 1.8, 2.5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        non_noise = labels != -1
        n_clusters = len(set(labels[non_noise]))
        if n_clusters < 2 or np.sum(non_noise) < max(3, int(0.4 * len(labels))):
            continue
        metrics = _clustering_metrics(X[non_noise], labels[non_noise])
        if metrics:
            candidates.append({
                "method": "DBSCAN",
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "labels": labels,
                "metrics": metrics,
                "n_outliers": int(np.sum(labels == -1)),
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
    labels = best.pop("labels")
    result = {
        "method": "Auto Best Clustering",
        "preprocessing": preprocessing_summary(df),
        "selected_method": best["method"],
        "n_clusters": int(best.get("n_clusters", len(np.unique(labels)))),
        "labels": [int(l) for l in labels],
        "cluster_sizes": _cluster_sizes(labels),
        "metrics": best["metrics"],
        "features_used": cols,
        "selection_reason": (
            f"Selected {best['method']} because it had the strongest silhouette score "
            f"among {len(candidates)} valid clustering candidates."
        ),
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
    }
    for key in ("eps", "min_samples", "n_outliers", "inertia", "linkage"):
        if key in best:
            result[key] = best[key]
    return result


# ── Anomaly Detection ──────────────────────────────────────────────────────

def detect_anomalies_isolation_forest(df: pd.DataFrame, contamination: float = 0.1) -> dict:
    """Isolation Forest anomaly detection."""
    X, cols = _numeric_matrix(df)
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X)
    scores = model.decision_function(X)

    n_outliers = int(np.sum(labels == -1))
    result = {
        "method": "Isolation Forest",
        "preprocessing": preprocessing_summary(df),
        "contamination": contamination,
        "n_outliers": n_outliers,
        "n_normal": int(np.sum(labels == 1)),
        "outlier_percentage": round(n_outliers / len(df) * 100, 2),
        "anomaly_indices": [int(i) for i in np.where(labels == -1)[0]],
        "anomaly_scores": [round(float(s), 4) for s in scores[labels == -1]],
    }
    
    # Add visualization data - scatter plot colored by anomaly status
    result["visualization"] = {
        "type": "scatter",
        "title": f"Isolation Forest Anomaly Detection ({n_outliers} anomalies)",
        "points": [
            {
                "x": round(float(X[i, 0]), 4),
                "y": round(float(X[i, 1]), 4),
                "cluster": -1 if labels[i] == -1 else 1  # -1 = anomaly, 1 = normal
            }
            for i in range(len(X))
        ],
        "x_label": cols[0] if len(cols) > 0 else "Feature 1",
        "y_label": cols[1] if len(cols) > 1 else "Feature 2",
    }
    
    return result


def detect_anomalies_lof(df: pd.DataFrame, contamination: float = 0.1, n_neighbors: int = 20) -> dict:
    """Local Outlier Factor anomaly detection."""
    X, cols = _numeric_matrix(df)
    model = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(X) - 1), contamination=contamination)
    labels = model.fit_predict(X)

    n_outliers = int(np.sum(labels == -1))
    result = {
        "method": "Local Outlier Factor",
        "preprocessing": preprocessing_summary(df),
        "n_neighbors": n_neighbors,
        "contamination": contamination,
        "n_outliers": n_outliers,
        "n_normal": int(np.sum(labels == 1)),
        "outlier_percentage": round(n_outliers / len(df) * 100, 2),
        "anomaly_indices": [int(i) for i in np.where(labels == -1)[0]],
    }
    
    # Add visualization data - scatter plot colored by anomaly status
    result["visualization"] = {
        "type": "scatter",
        "title": f"Local Outlier Factor Anomaly Detection ({n_outliers} anomalies)",
        "points": [
            {
                "x": round(float(X[i, 0]), 4),
                "y": round(float(X[i, 1]), 4),
                "cluster": -1 if labels[i] == -1 else 1  # -1 = anomaly, 1 = normal
            }
            for i in range(len(X))
        ],
        "x_label": cols[0] if len(cols) > 0 else "Feature 1",
        "y_label": cols[1] if len(cols) > 1 else "Feature 2",
    }
    
    return result


# ── Dimensionality Reduction ───────────────────────────────────────────────

def reduce_tsne(df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0) -> dict:
    """t-SNE dimensionality reduction for visualization."""
    X, cols = _numeric_matrix(df)
    perplexity = min(perplexity, len(X) - 1)
    model = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    embedding = model.fit_transform(X)

    result = {
        "method": "t-SNE",
        "preprocessing": preprocessing_summary(df),
        "n_components": n_components,
        "perplexity": perplexity,
        "points": [
            {"x": round(float(embedding[i, 0]), 4), "y": round(float(embedding[i, 1]), 4), "index": i}
            for i in range(len(embedding))
        ],
        "visualization": {
            "type": "scatter",
            "title": f"t-SNE Projection (perplexity={perplexity})",
            "points": [
                {"x": round(float(embedding[i, 0]), 4), "y": round(float(embedding[i, 1]), 4), "cluster": 0}
                for i in range(len(embedding))
            ],
            "x_label": "t-SNE 1",
            "y_label": "t-SNE 2",
        },
    }
    return result


def reduce_pca_advanced(df: pd.DataFrame, n_components: int = 2) -> dict:
    """Advanced PCA with variance explained."""
    X, cols = _numeric_matrix(df)
    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(X)

    explained_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]

    result = {
        "method": "PCA",
        "preprocessing": preprocessing_summary(df),
        "n_components": n_components,
        "explained_variance": explained_var,
        "cumulative_variance": [round(float(v), 2) for v in np.cumsum(pca.explained_variance_ratio_)],
        "points": [
            {"x": round(float(embedding[i, 0]), 4), "y": round(float(embedding[i, 1]), 4), "index": i}
            for i in range(len(embedding))
        ],
        "visualization": {
            "type": "scatter",
            "title": f"PCA Projection (PC1 vs PC2)",
            "points": [
                {"x": round(float(embedding[i, 0]), 4), "y": round(float(embedding[i, 1]), 4), "cluster": 0}
                for i in range(len(embedding))
            ],
            "x_label": f"PC1 ({explained_var[0]}%)",
            "y_label": f"PC2 ({explained_var[1]}%)",
        },
    }
    return result


# ── Association Rules ──────────────────────────────────────────────────────

def association_rules(df: pd.DataFrame, min_support: float = 0.1, min_confidence: float = 0.5) -> dict:
    """Simple frequent itemset mining for categorical/boolean columns."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules as ar_rules
    except ImportError:
        raise ImportError("mlxtend is required for association rules. Install: pip install mlxtend")

    # Convert categorical columns to boolean one-hot encoding
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        raise ValueError("Need at least one categorical/boolean column for association rules.")

    df_encoded = pd.get_dummies(df[cat_cols], prefix_sep="=", dummy_na=False)

    # Filter to binary columns only
    binary_cols = [c for c in df_encoded.columns if df_encoded[c].nunique() == 2]
    if not binary_cols:
        raise ValueError("Could not create binary columns from categorical data.")

    df_binary = df_encoded[binary_cols].astype(bool)

    # Find frequent itemsets
    frequent_itemsets = apriori(df_binary, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        return {
            "method": "Apriori",
            "min_support": min_support,
            "min_confidence": min_confidence,
            "n_rules": 0,
            "rules": [],
            "message": f"No frequent itemsets found with min_support={min_support}. Try lowering it.",
        }

    # Generate association rules
    rules = ar_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        return {
            "method": "Apriori",
            "min_support": min_support,
            "min_confidence": min_confidence,
            "n_rules": 0,
            "rules": [],
            "message": f"No rules found with min_confidence={min_confidence}. Try lowering it.",
        }

    # Format rules
    formatted_rules = []
    for _, row in rules.iterrows():
        formatted_rules.append({
            "antecedents": list(row["antecedents"]),
            "consequents": list(row["consequents"]),
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 4),
        })

    formatted_rules.sort(key=lambda x: x["lift"], reverse=True)

    return {
        "method": "Apriori",
        "min_support": min_support,
        "min_confidence": min_confidence,
        "n_rules": len(formatted_rules),
        "rules": formatted_rules[:50],  # Limit to top 50
    }


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
