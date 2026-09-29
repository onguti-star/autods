"""Large-dataset behaviour for the unsupervised module.

AutoDS accepts frames up to 5.5M rows, but the unsupervised algorithms are all
"fit the whole matrix" methods whose cost grows badly with row count. These
tests pin the row budgets in unsupervised.py so a big run returns a correct
answer on a sample instead of hanging the request.
"""
import time

import numpy as np
import pandas as pd
import pytest

from backend import unsupervised as us


def _frame(n_rows: int, n_numeric: int = 4, seed: int = 0) -> pd.DataFrame:
    """A frame with real cluster structure, so metrics are meaningful."""
    rng = np.random.default_rng(seed)
    data = {"target": rng.integers(0, 2, size=n_rows)}
    for i in range(n_numeric):
        # Three well-separated groups, so a correct clustering is findable.
        data[f"num{i}"] = rng.normal(loc=(rng.integers(0, 3, size=n_rows) * 5), scale=1.0)
    return pd.DataFrame(data)


def test_subsample_is_deterministic_sorted_and_capped():
    first = us._subsample_indices(10_000, 500)
    second = us._subsample_indices(10_000, 500)

    assert len(first) == 500
    # Same seed -> same sample, so repeated runs on one dataset agree.
    assert np.array_equal(first, second)
    # Sorted, so the sample keeps the frame's row order.
    assert np.array_equal(first, np.sort(first))
    # Under the cap means no sampling at all.
    assert np.array_equal(us._subsample_indices(100, 500), np.arange(100))


def test_kmeans_uses_minibatch_solver_above_the_full_row_threshold():
    assert type(us._make_kmeans(3, 1_000)).__name__ == "KMeans"
    assert type(us._make_kmeans(3, us.KMEANS_FULL_ROWS + 1)).__name__ == "MiniBatchKMeans"


def test_kmeans_labels_every_row_on_a_large_frame():
    """The fit is capped, but predict() must still label every row.

    This is the property that keeps the cluster_label column working: the model
    is cheap to fit on a sample, and assigning the rest is a nearest-centroid
    lookup rather than a refit. 120k rows is under KMEANS_MAX_FIT_ROWS, so this
    checks the MiniBatchKMeans switch on its own.
    """
    df = _frame(120_000)
    result = us.cluster_kmeans(df, n_clusters=3)

    assert result["n_rows_scored"] == 120_000
    assert len(result["labels"]) == 120_000
    assert result["labels_complete"] is True
    assert result["kmeans_solver"] == "MiniBatchKMeans"
    # Under the fit budget, so no rows were dropped.
    assert result["scaling"]["sampled"] is False
    assert sum(result["cluster_sizes"].values()) == 120_000
    assert "inertia_note" not in result


def test_kmeans_samples_the_fit_above_the_row_budget():
    """Past KMEANS_MAX_FIT_ROWS the fit is sampled, and that is reported."""
    df = _frame(us.KMEANS_MAX_FIT_ROWS + 20_000, n_numeric=2)
    result = us.cluster_kmeans(df, n_clusters=3)

    assert result["scaling"]["sampled"] is True
    assert result["scaling"]["rows_used"] == us.KMEANS_MAX_FIT_ROWS
    assert "note" in result["scaling"]
    # Sampling the fit does not shrink the analysed set -- predict() still
    # scored every row, and n_rows_scored stays exact even though the returned
    # label list is capped to keep the JSON small.
    assert result["n_rows_scored"] == len(df)
    assert result["labels_complete"] is False
    assert len(result["labels"]) == us.MAX_RETURNED_LABELS
    assert "labels_note" in result
    assert "inertia_note" in result


def test_kmeans_completes_quickly_on_a_large_frame():
    df = _frame(200_000)
    start = time.time()
    result = us.cluster_kmeans(df, n_clusters=4)
    elapsed = time.time() - start

    assert result["n_rows_scored"] == 200_000
    assert elapsed < 60, f"K-Means took {elapsed:.1f}s on 200k rows -- expected < 60s"


def test_clustering_metrics_are_scored_on_a_bounded_sample():
    """Full-data scoring is what used to stall; metrics must report the cap."""
    df = _frame(60_000)
    result = us.cluster_kmeans(df, n_clusters=3)

    metrics = result["metrics"]
    assert metrics["scored_on_rows"] == us.METRICS_MAX_ROWS
    assert metrics["scored_on_sample"] is True
    # Still a real score, just an approximate one.
    assert 0.0 < metrics["silhouette"] <= 1.0


def test_suggest_clusters_completes_on_a_large_frame():
    """One fit per K value used to mean nine full-data KMeans runs."""
    df = _frame(150_000)
    start = time.time()
    result = us.suggest_clusters(df, max_clusters=10)
    elapsed = time.time() - start

    assert result["k_values"] == list(range(2, 11))
    assert result["recommended_k"] in result["k_values"]
    assert result["scaling"]["rows_used"] == us.SUGGEST_MAX_ROWS
    assert len(result["inertias"]) == len(result["k_values"])
    assert elapsed < 90, f"suggest_clusters took {elapsed:.1f}s -- expected < 90s"


def test_cluster_best_searches_small_then_refits_the_winner():
    """Auto Best fits ~26 models, so it searches a sample and refits the winner."""
    df = _frame(150_000)
    start = time.time()
    result = us.cluster_best(df, max_clusters=8)
    elapsed = time.time() - start

    assert result["selected_method"] in {"K-Means", "Hierarchical", "DBSCAN"}
    assert result["candidates"], "expected the candidate leaderboard"
    assert len(result["labels"]) > 0
    assert result["labels_complete"] is True
    assert elapsed < 120, f"cluster_best took {elapsed:.1f}s -- expected < 120s"


def test_hierarchical_refuses_large_frames_with_a_useful_message():
    """Agglomerative needs an n x n matrix, so it must fail fast and clearly."""
    df = _frame(us.HIERARCHICAL_MAX_ROWS + 1_000)
    with pytest.raises(ValueError) as exc:
        us.cluster_hierarchical(df, n_clusters=3)

    message = str(exc.value)
    assert f"{us.HIERARCHICAL_MAX_ROWS:,}" in message
    assert "K-Means" in message, "the error should point at a workable alternative"


def test_dbscan_is_capped_and_reports_the_sample():
    df = _frame(80_000)
    result = us.cluster_dbscan(df, eps=0.5, min_samples=5)

    assert result["scaling"]["sampled"] is True
    assert result["n_rows_scored"] == us.DBSCAN_MAX_ROWS
    # Outlier share is relative to the rows actually clustered.
    assert 0.0 <= result["outlier_percentage"] <= 100.0
    assert sum(result["cluster_sizes"].values()) == us.DBSCAN_MAX_ROWS


def test_dbscan_outlier_percentage_is_relative_to_clustered_rows():
    df = _frame(80_000)
    result = us.cluster_dbscan(df, eps=0.5, min_samples=5)

    expected = round(result["n_outliers"] / us.DBSCAN_MAX_ROWS * 100, 2)
    assert result["outlier_percentage"] == expected
    # Reported against the full frame this would look like almost no outliers.
    assert result["outlier_percentage"] >= result["n_outliers"] / len(df) * 100


def test_lof_is_capped_and_maps_indices_back_to_real_rows():
    """LOF labels a sampled matrix, so the indices must be real row numbers."""
    df = _frame(60_000)
    result = us.detect_anomalies_lof(df, contamination=0.05, n_neighbors=20)

    assert result["scaling"]["rows_used"] == us.LOF_MAX_ROWS
    assert result["n_outliers"] > 0
    assert all(0 <= i < len(df) for i in result["anomaly_indices"])


def test_isolation_forest_scores_every_row_on_a_large_frame():
    df = _frame(150_000)
    result = us.detect_anomalies_isolation_forest(df, contamination=0.05)

    assert result["n_outliers"] > 0
    assert result["n_outliers"] + result["n_normal"] == 150_000
    assert result["scaling"]["sampled"] is False
    assert all(0 <= i < len(df) for i in result["anomaly_indices"])


def test_anomaly_index_list_is_capped_but_the_count_is_exact():
    """contamination=0.5 on 200k rows is 100k indices of pure JSON overhead."""
    df = _frame(200_000)
    result = us.detect_anomalies_isolation_forest(df, contamination=0.5)

    assert result["n_outliers"] > us.MAX_RETURNED_ANOMALY_INDICES
    assert len(result["anomaly_indices"]) == us.MAX_RETURNED_ANOMALY_INDICES
    assert result["anomaly_indices_truncated"] is True
    # The count is the number users act on, so it must not be capped.
    assert result["n_normal"] + result["n_outliers"] == 200_000
    assert result["outlier_percentage"] > 0


def test_scatter_payload_is_capped_for_every_charted_method():
    """A 100k-point canvas is unusable in the browser, so charts get capped."""
    df = _frame(100_000)
    results = [
        us.cluster_kmeans(df, n_clusters=3),
        us.cluster_dbscan(df),
        us.detect_anomalies_isolation_forest(df, contamination=0.05),
        us.reduce_pca_advanced(df),
    ]
    for result in results:
        points = result["visualization"]["points"]
        assert len(points) <= us.MAX_SCATTER_POINTS
        assert result["visualization"]["n_points_total"] > us.MAX_SCATTER_POINTS
        assert "note" in result["visualization"]


def test_labels_payload_is_capped_so_json_cannot_stall():
    labels = np.arange(us.MAX_RETURNED_LABELS + 5_000) % 4
    payload = us._labels_payload(labels)

    assert len(payload["labels"]) == us.MAX_RETURNED_LABELS
    assert payload["labels_complete"] is False
    assert payload["n_rows_scored"] == len(labels)
    assert "labels_note" in payload


def test_tsne_is_capped_and_pca_pre_reduces_wide_frames():
    df = _frame(40_000, n_numeric=60)
    start = time.time()
    result = us.reduce_tsne(df, n_components=2, perplexity=30.0)
    elapsed = time.time() - start

    assert result["scaling"]["rows_used"] == us.TSNE_MAX_ROWS
    # The wide frame was compressed before t-SNE ran.
    assert "preprocessing_note" in result
    # Point indices refer to real rows in the original frame.
    assert all(0 <= p["index"] < len(df) for p in result["points"])
    assert elapsed < 120, f"t-SNE took {elapsed:.1f}s on 40k rows -- expected < 120s"


def test_pca_uses_the_randomized_solver_on_large_frames():
    """The exact SVD is the expensive part of PCA on a big matrix."""
    df = _frame(50_000, n_numeric=8)
    result = us.reduce_pca_advanced(df, n_components=2)

    assert result["solver"] == "randomized"
    assert result["n_rows_projected"] == 50_000
    assert len(result["points"]) == us.MAX_SCATTER_POINTS
    assert sum(result["explained_variance"]) > 0


def test_association_rules_skip_high_cardinality_columns():
    """One-hot encoding an ID column is the classic Apriori hang."""
    rng = np.random.default_rng(0)
    n_rows = 20_000
    df = pd.DataFrame({
        "region": rng.choice(["north", "south", "east"], size=n_rows),
        "tier": rng.choice(["basic", "premium"], size=n_rows),
        # Nearly unique: can never form a frequent itemset.
        "customer_id": [f"id_{i}" for i in range(n_rows)],
    })
    result = us.association_rules(df, min_support=0.1, min_confidence=0.2)

    scaling = result["scaling"]
    dropped = [c["column"] for c in scaling["dropped_high_cardinality_columns"]]
    assert "customer_id" in dropped
    assert scaling["dropped_note"]
    assert result["max_itemset_len"] == us.ASSOCIATION_MAX_ITEMSET_LEN
    assert result["n_transactions"] == len(df)
    assert all("customer_id" not in r["antecedents"] for r in result["rules"])


def test_association_rules_reject_an_all_unique_frame_clearly():
    df = pd.DataFrame({"free_text": [f"row {i} text" for i in range(500)]})
    with pytest.raises(ValueError) as exc:
        us.association_rules(df, min_support=0.1, min_confidence=0.5)

    assert "distinct values" in str(exc.value)


def test_small_dataset_results_are_unchanged_in_shape():
    """The caps must not disturb the normal small-data path."""
    df = _frame(500)
    result = us.cluster_kmeans(df, n_clusters=3)

    assert result["labels_complete"] is True
    assert result["scaling"]["sampled"] is False
    assert "inertia_note" not in result
    assert len(result["labels"]) == 500
    assert result["kmeans_solver"] == "KMeans"
    assert result["metrics"]["scored_on_sample"] is False
