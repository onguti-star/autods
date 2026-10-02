import re
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


def test_html_report_auto_includes_charts_when_none_were_staged():
    # Regression test: if the person downloads the report without having
    # visited the Visuals tab (so no chart history was ever captured), the
    # report used to ship with zero charts and no indication why. It
    # should now fall back to auto-suggested charts instead.
    df = pd.DataFrame({
        "price": list(range(1, 31)),
        "quantity": list(range(30, 0, -1)),
        "category": (["A", "B", "C"] * 10),
    })

    report = _build_html_report(_session_for_report(df))

    m = re.search(r"const reportCharts = (\[.*?\]);", report, re.S)
    assert m is not None
    assert m.group(1).strip() != "[]"


def test_cleaning_log_appears_before_visualizations_not_after():
    # Regression test: the cleaning log used to render after the
    # Visualizations/PCA sections, which reads oddly — cleaning happens
    # first in the real workflow, so it should appear earlier in the report.
    df = pd.DataFrame({
        "price": list(range(1, 31)),
        "quantity": list(range(30, 0, -1)),
    })
    session = _session_for_report(df)
    session.cleaning_log = ["Removed 2 duplicate row(s)."]

    report = _build_html_report(session)

    cleaning_pos = report.index('id="cleaning"')
    summary_pos = report.index('id="summary"')
    viz_pos = report.index('id="visualizations"')

    assert cleaning_pos < summary_pos
    assert cleaning_pos < viz_pos
