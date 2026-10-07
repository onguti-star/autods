import pandas as pd
import pytest

from backend import viz


def _sample_df():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West"] * 10,
        "product": ["Widget", "Gadget"] * 20,
    })


def test_stacked_bar_builds_one_series_per_group_value():
    df = _sample_df()

    chart = viz.chart_data(df, "region", "stacked_bar", group="product")

    assert chart["type"] == "stacked_bar"
    assert sorted(chart["labels"]) == ["East", "North", "South", "West"]
    series_labels = sorted(s["label"] for s in chart["series"])
    assert series_labels == ["Gadget", "Widget"]
    # Every x-category's segment values should sum to its row count (10 each)
    for s in chart["series"]:
        assert sum(s["values"]) == 20


def test_stacked_bar_requires_a_group_column():
    df = _sample_df()

    with pytest.raises(ValueError, match="Group by"):
        viz.chart_data(df, "region", "stacked_bar")


def test_stacked_bar_rejects_unknown_group_column():
    df = _sample_df()

    with pytest.raises(ValueError, match="not found"):
        viz.chart_data(df, "region", "stacked_bar", group="nonexistent")


def test_stacked_bar_caps_x_categories_via_bar_limit():
    df = pd.DataFrame({
        "category": [f"cat_{i}" for i in range(50) for _ in range(2)],
        "status": ["active", "inactive"] * 50,
    })

    chart = viz.chart_data(df, "category", "stacked_bar", group="status", bar_limit=5)

    assert len(chart["labels"]) == 5
    assert "top 5 of 50" in chart["caption"]


def test_stacked_bar_folds_excess_group_values_into_other():
    # 12 distinct product values, well beyond the default max_series cap (8)
    df = pd.DataFrame({
        "region": ["North", "South"] * 60,
        "product": [f"product_{i % 12}" for i in range(120)],
    })

    chart = viz.chart_data(df, "region", "stacked_bar", group="product")

    series_labels = [s["label"] for s in chart["series"]]
    assert len(series_labels) <= 8
    assert "Other" in series_labels
