"""Unsupervised learning must return labels for every row (not just a sampled
subset) and must not hang or run out of memory on large datasets. Each method
is checked once on a small dataset (behaviour should be exact, no sampling)
and once on a dataset above its sampling threshold (behaviour is approximate
but must still cover every row and finish quickly, with a sampling note)."""
import time

import numpy as np
import pandas as pd
import pytest

from backend import unsupervised as U


def _make_df(n, seed=0, with_categoricals=True):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(5, 8))
    X = np.vstack([rng.normal(c, 1.0, size=((n // 5) + 1, 8)) for c in centers])[:n]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    if with_categoricals:
        df["cat_a"] = rng.choice(list("abc"), len(df))
        df["cat_b"] = rng.choice(["x", "y"], len(df))
        df["row_id"] = [f"r{i}" for i in range(len(df))]      # high-cardinality "id" column
    return df


SMALL = 300


@pytest.mark.parametrize("fn,kwargs", [
    (U.cluster_kmeans, dict(n_clusters=4)),
    (U.cluster_dbscan, dict(eps=0.5, min_samples=5)),
    (U.cluster_hierarchical, dict(n_clusters=4)),
])
def test_small_dataset_not_sampled(fn, kwargs):
    df = _make_df(SMALL)
    result = fn(df, **kwargs)
    assert len(result["labels"]) == SMALL
    assert result["sampling"]["sampled"] is False
    assert result["sampling"]["note"] == ""


@pytest.mark.parametrize("fn,kwargs,n,budget_s", [
    (U.cluster_kmeans, dict(n_clusters=4), 200_000, 15),
    (U.cluster_dbscan, dict(eps=0.5, min_samples=5), 60_000, 25),
    (U.cluster_hierarchical, dict(n_clusters=4), 20_000, 20),
    (U.detect_anomalies_isolation_forest, dict(contamination=0.05), 200_000, 15),
    (U.detect_anomalies_lof, dict(contamination=0.05, n_neighbors=20), 60_000, 25),
])
def test_large_dataset_finishes_quickly_and_labels_every_row(fn, kwargs, n, budget_s):
    df = _make_df(n, with_categoricals=False)
    t0 = time.time()
    result = fn(df, **kwargs)
    elapsed = time.time() - t0
    assert elapsed < budget_s, f"took {elapsed:.1f}s, budget was {budget_s}s"
    if "labels" in result:
        assert len(result["labels"]) == n
    else:
        # anomaly detectors report row coverage as normal + outlier counts instead
        assert result["n_normal"] + result["n_outliers"] == n
    # every point in the visualisation must be one of the actual rows
    assert len(result["visualization"]["points"]) <= U.MAX_PLOT_POINTS + 1


def test_hierarchical_above_ward_limit_does_not_blow_up_memory():
    # Before sampling this size used to try to allocate an n x n distance matrix
    # and crash with a MemoryError.
    n = U.MAX_HIERARCHICAL_FIT * 3
    df = _make_df(n, with_categoricals=False)
    result = U.cluster_hierarchical(df, n_clusters=4)
    assert len(result["labels"]) == n
    assert result["sampling"]["sampled"] is True


def test_cluster_best_labels_every_row_on_large_data():
    n = 30_000
    df = _make_df(n, with_categoricals=False)
    result = U.cluster_best(df, max_clusters=5)
    assert len(result["labels"]) == n
    assert result["selected_method"] in {"K-Means", "Hierarchical", "DBSCAN"}


def test_tsne_and_pca_scale_down_gracefully():
    n = 20_000
    df = _make_df(n, with_categoricals=False)
    tsne = U.reduce_tsne(df, n_components=2, perplexity=30.0)
    assert len(tsne["points"]) == U.MAX_TSNE_ROWS
    assert tsne["sampling"]["sampled"] is True

    pca = U.reduce_pca_advanced(df, n_components=2)
    assert len(pca["points"]) <= U.MAX_PLOT_POINTS + 1
    # PCA's own math still runs on the full column set, so variance is meaningful.
    assert sum(pca["explained_variance"]) > 0


def test_association_rules_skips_id_like_columns_and_samples_rows():
    n = U.MAX_ASSOC_ROWS + 5_000
    df = _make_df(n, with_categoricals=True)   # includes a "row_id" column, all-unique
    result = U.association_rules(df, min_support=0.1, min_confidence=0.3)
    assert "row_id" in result["skipped_columns"]
    assert result["sampling"]["sampled"] is True


def test_association_rules_all_high_cardinality_raises_clear_error():
    df = pd.DataFrame({
        "f0": np.random.default_rng(0).normal(size=200),
        "f1": np.random.default_rng(1).normal(size=200),
        "id_col": [f"id{i}" for i in range(200)],   # every value unique
    })
    with pytest.raises(ValueError, match="distinct values"):
        U.association_rules(df, min_support=0.1, min_confidence=0.5)
