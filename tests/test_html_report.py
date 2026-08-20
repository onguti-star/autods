from types import SimpleNamespace

import pandas as pd

from backend.main_report import _build_html_report


def _session_for_report(df: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        df=df,
        filename="sales.csv",
        cleaning_log=[],
        chat_clean_log=[
            {"command": "create a new column called revenue", "message": "Created empty column 'revenue'."},
            {"command": "fill revenue with price * quantity", "message": "Filled column 'revenue' with expression: price * quantity"},
        ],
        pca_result={},
        feature_importance=[],
        best_model_name=None,
        models={},
        target=None,
        leaderboard=[],
        saved_runs={},
        saved_predictions={},
        unsupervised_results={},
        feature_columns=[],
        problem_type=None,
    )


def test_html_report_includes_quality_details_and_clean_assist_column_workflow():
    df = pd.DataFrame({
        "price": [10, 20, 20],
        "quantity": [2, 3, 3],
        "revenue": [20, 60, 60],
        "notes": ["a", None, None],
    })

    report = _build_html_report(_session_for_report(df))

    assert "Data Quality Details" in report
    assert "Missing Cell Rate" in report
    assert "Numeric Column Ranges" in report
    assert "Clean Assist commands" in report
    assert "Created empty column" in report
    assert "Derived column" in report
    assert "fill revenue with price * quantity" in report
