import numpy as np
import pandas as pd

from backend import automl


def test_large_dataset_sample_size_scales_and_caps():
    # Stale expectations here (20_000 / 50_001 / 75_000) drifted out of sync with
    # LARGE_DATASET_MIN_SAMPLE/MAX_SAMPLE/FRACTION in automl.py and were silently
    # failing on every run -- updated to match the current, intended constants
    # (min 30k, max 150k, 8% adaptive fraction in between).
    assert automl._large_dataset_sample_size(120_000) == 30_000
    assert automl._large_dataset_sample_size(1_000_001) == 80_001
    assert automl._large_dataset_sample_size(2_000_000) == 150_000


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

    problem_type, leaderboard, fitted, best_name, label_encoder = automl.train_all(
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

    problem_type, leaderboard, fitted, best_name, label_encoder = automl.train_all(
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
    _, _, fitted, best_name, _ = automl.train_all(df, "target", progress_callback=lambda _: None)
    # No X argument passed -- feature names/importances come entirely from the
    # already-fitted pipeline, so callers should not need to hand over the
    # (potentially huge) training frame just to read this off.
    importance = automl.feature_importance(fitted[best_name])
    assert isinstance(importance, list)
