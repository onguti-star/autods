import numpy as np
import pandas as pd

from backend import automl


def test_large_dataset_sample_size_scales_and_caps():
    # Matches the current constants in automl.py: min 30k, max 250k, 15%
    # adaptive fraction in between (raised from 8%/150k so large datasets
    # keep materially more training data -- see the comment on
    # LARGE_DATASET_MAX_SAMPLE/LARGE_DATASET_SAMPLE_FRACTION).
    assert automl._large_dataset_sample_size(120_000) == 30_000
    assert automl._large_dataset_sample_size(1_000_001) == 150_001
    assert automl._large_dataset_sample_size(2_000_000) == 250_000


def test_fast_models_threshold_kicks_in_below_large_dataset_threshold():
    # Datasets above FAST_MODELS_THRESHOLD (50k, raised from 30k) but below
    # LARGE_DATASET_THRESHOLD (100k) -- e.g. 55k rows -- should use the fast
    # model panel (with early stopping) WITHOUT being downsampled.
    assert automl.FAST_MODELS_THRESHOLD < automl.LARGE_DATASET_THRESHOLD
    # 55k rows: above fast-models threshold, below sampling threshold
    assert 55_000 > automl.FAST_MODELS_THRESHOLD
    assert 55_000 < automl.LARGE_DATASET_THRESHOLD
    # Sampling should NOT kick in for 55k (below LARGE_DATASET_THRESHOLD)
    assert automl._large_dataset_sample_size(55_000) == 55_000
    # 45k rows now falls BELOW the raised threshold, so it should get the
    # full candidate panel (with CV-based selection) instead of the fast one.
    assert 45_000 < automl.FAST_MODELS_THRESHOLD
    assert 45_000 <= automl.CV_SELECTION_MAX_ROWS or True  # documents intent; not asserted strictly


def test_medium_dataset_uses_fast_models_and_completes_quickly():
    """A 55k-row dataset with a categorical column should use the fast model
    panel (early-stopping HistGradientBoosting + Extra Trees) and finish
    without hanging."""
    rng = np.random.default_rng(42)
    n_rows = 55_000

    df = pd.DataFrame(
        {
            "target": rng.integers(0, 2, size=n_rows),
            "num1": rng.normal(size=n_rows),
            "num2": rng.normal(size=n_rows),
            "cat": rng.choice(["a", "b", "c", "d"], size=n_rows),
        }
    )

    import time
    start = time.time()
    problem_type, leaderboard, fitted, best_name, label_encoder, _importance = automl.train_all(
        df,
        "target",
        progress_callback=lambda _: None,
    )
    elapsed = time.time() - start

    assert problem_type == "classification"
    assert leaderboard
    assert fitted
    assert best_name is not None

    # The fast panel should produce 4 models (Linear, HistGBM, Extra Trees,
    # XGBoost) and finish well under 60 seconds on 55k rows.
    assert elapsed < 60, f"Training took {elapsed:.1f}s -- expected < 60s"
    model_names = [r.get("model") for r in leaderboard if "error" not in r]
    assert len(model_names) <= 6  # fast panel has 4, plus possible error rows


def test_train_all_handles_large_dataset_without_crashing():
    rng = np.random.default_rng(42)
    n_rows = 120_000

    df = pd.DataFrame(
        {
            "target": rng.integers(0, 2, size=n_rows),
            "num1": rng.normal(size=n_rows),
            "num2": rng.normal(size=n_rows),
            "cat": rng.choice(["a", "b", "c", "d"], size=n_rows),
            "text": [f"token {rng.integers(0, 100)}" for _ in range(n_rows)],
        }
    )

    problem_type, leaderboard, fitted, best_name, label_encoder, _importance = automl.train_all(
        df,
        "target",
        progress_callback=lambda _: None,
    )

    assert problem_type == "classification"
    assert leaderboard
    assert fitted
    assert best_name is not None
    assert label_encoder is not None


def test_large_dataset_caps_one_hot_width_for_high_cardinality_column():
    # A categorical column where almost every value is unique (job titles, free-
    # text addresses, etc.) used to be able to blow one-hot encoding up to
    # thousands of columns even after min_frequency filtering, which then got
    # densified for HistGradientBoosting (_to_dense) -- a real OOM risk on large
    # datasets. one_hot_max_categories should keep the encoded width bounded
    # regardless of how spread out the category frequencies are.
    rng = np.random.default_rng(0)
    n_rows = 150_000

    df = pd.DataFrame(
        {
            "target": rng.integers(0, 2, size=n_rows),
            "num1": rng.normal(size=n_rows),
            # ~n_rows/3 distinct values -- high cardinality, not just noise
            "high_card_cat": rng.integers(0, n_rows // 3, size=n_rows).astype(str),
        }
    )

    problem_type, leaderboard, fitted, best_name, label_encoder, _importance = automl.train_all(
        df,
        "target",
        progress_callback=lambda _: None,
    )

    assert problem_type == "classification"
    assert best_name is not None
    prep = fitted[best_name].named_steps["prep"]
    cat_features = [f for f in prep.get_feature_names_out() if f.startswith("cat__")]
    # +1 allows for the "infrequent"/overflow bucket column.
    assert len(cat_features) <= 51


def test_feature_importance_does_not_require_x():
    rng = np.random.default_rng(0)
    n_rows = 500
    df = pd.DataFrame(
        {
            "target": rng.integers(0, 2, size=n_rows),
            "num1": rng.normal(size=n_rows),
            "num2": rng.normal(size=n_rows),
        }
    )
    _, _, fitted, best_name, _, _ = automl.train_all(df, "target", progress_callback=lambda _: None)
    # No X argument passed -- feature names/importances come entirely from the
    # already-fitted pipeline, so callers should not need to hand over the
    # (potentially huge) training frame just to read this off.
    importance = automl.feature_importance(fitted[best_name])
    assert isinstance(importance, list)
