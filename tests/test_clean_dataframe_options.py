import numpy as np
import pandas as pd

from backend.clean import clean_dataframe


def _dataset_with_everything_to_clean():
    """A dataframe with one issue for every Clean Now option: duplicate
    rows, a constant column, a >50%-missing column, an outlier, and a text
    column with whitespace that's been downcast to category dtype (the way
    low-cardinality text columns come back from the upload pipeline)."""
    df = pd.DataFrame({
        "id": range(1, 11),
        "constant_col": [1] * 10,
        "mostly_empty": [np.nan] * 6 + [1, 2, 3, 4],
        "score": [50, 51, 49, 52, 48, 50, 51, 49, 50, 500],  # 500 is an outlier
        "name": ["  Alice  ", "Bob"] * 5,
    })
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate row
    df["name"] = df["name"].astype("category")  # mimic upload-time downcasting
    return df


def test_unchecked_options_are_not_applied():
    df = _dataset_with_everything_to_clean()
    opts = {
        "drop_duplicates": False,
        "drop_constant_cols": False,
        "drop_high_missing_cols": False,
        "remove_outliers": False,
        "strip_whitespace": True,
        "fix_mixed_types": True,
        "fix_column_names": True,
        "fill_missing_numeric": "none",
        "fill_missing_categorical": "none",
    }
    out, log = clean_dataframe(df, opts)

    # Unchecked structural operations must leave the data untouched.
    assert len(out) == len(df)  # duplicate row still present
    assert "constant_col" in out.columns
    assert "mostly_empty" in out.columns
    assert out["score"].max() == 500  # outlier not clipped

    log_text = " ".join(log)
    assert "duplicate" not in log_text.lower()
    assert "constant" not in log_text.lower()
    assert "missing values" not in log_text.lower() or "50" not in log_text


def test_checked_whitespace_strip_applies_to_category_dtype_columns():
    # Regression test: low-cardinality text columns get downcast to
    # "category" dtype on upload, which used to make strip_whitespace
    # silently skip them even when checked.
    df = _dataset_with_everything_to_clean()
    assert isinstance(df["name"].dtype, pd.CategoricalDtype)

    out, log = clean_dataframe(df, {
        "strip_whitespace": True,
        "drop_duplicates": False,
        "drop_constant_cols": False,
        "drop_high_missing_cols": False,
        "remove_outliers": False,
        "fix_mixed_types": False,
        "fix_column_names": False,
        "fill_missing_numeric": "none",
        "fill_missing_categorical": "none",
    })

    assert out["name"].astype(str).tolist()[0] == "Alice"
    assert any("whitespace" in line.lower() for line in log)


def test_checked_fill_missing_categorical_applies_to_category_dtype_columns():
    df = pd.DataFrame({"status": ["active", "active", None, "closed"]}).astype(
        {"status": "category"}
    )

    out, log = clean_dataframe(df, {
        "fill_missing_categorical": "mode",
        "drop_duplicates": False,
        "drop_constant_cols": False,
        "drop_high_missing_cols": False,
        "remove_outliers": False,
        "strip_whitespace": False,
        "fix_mixed_types": False,
        "fix_column_names": False,
        "fill_missing_numeric": "none",
    })

    assert out["status"].isna().sum() == 0
    assert any("missing categorical" in line.lower() for line in log)


def test_all_options_checked_applies_every_step():
    df = _dataset_with_everything_to_clean()
    out, log = clean_dataframe(df, {
        "drop_duplicates": True,
        "drop_constant_cols": True,
        "drop_high_missing_cols": True,
        "remove_outliers": True,
        "strip_whitespace": True,
        "fix_mixed_types": True,
        "fix_column_names": True,
        "fill_missing_numeric": "median",
        "fill_missing_categorical": "mode",
    })

    assert len(out) == len(df) - 1  # duplicate row removed
    assert "constant_col" not in out.columns
    assert "mostly_empty" not in out.columns
    assert out["score"].max() < 500  # outlier clipped
    assert out["name"].astype(str).tolist()[0] == "Alice"  # whitespace stripped
    assert len(log) >= 4
