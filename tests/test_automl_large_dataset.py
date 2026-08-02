import numpy as np
import pandas as pd

from backend import automl


def test_large_dataset_sample_size_scales_and_caps():
    assert automl._large_dataset_sample_size(120_000) == 20_000
    assert automl._large_dataset_sample_size(1_000_001) == 50_001
    assert automl._large_dataset_sample_size(2_000_000) == 75_000


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
