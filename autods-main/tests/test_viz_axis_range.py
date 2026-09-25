import pandas as pd
import pytest

from backend import viz


def test_histogram_respects_x_axis_range():
    df = pd.DataFrame({"score": [1, 2, 3, 4, 5, 100]})

    chart = viz.chart_data(df, "score", "histogram", x_min=2, x_max=5)

    assert chart["x_min"] == 2
    assert chart["x_max"] == 5
    assert sum(chart["values"]) == 4


def test_histogram_rejects_backwards_x_axis_range():
    df = pd.DataFrame({"score": [1, 2, 3]})

    with pytest.raises(ValueError, match="minimum"):
        viz.chart_data(df, "score", "histogram", x_min=5, x_max=2)


def test_histogram_respects_custom_bin_width():
    df = pd.DataFrame({"score": [0, 5, 19, 20, 21, 39, 40]})

    chart = viz.chart_data(df, "score", "histogram", x_min=0, x_max=60, bin_width=20)

    assert chart["bin_width"] == 20
    assert chart["labels"] == ["0.00–20.00", "20.00–40.00", "40.00–60.00"]
    assert chart["values"] == [3, 3, 1]


def test_histogram_rejects_invalid_bin_width():
    df = pd.DataFrame({"score": [1, 2, 3]})

    with pytest.raises(ValueError, match="greater than zero"):
        viz.chart_data(df, "score", "histogram", bin_width=0)


def test_bar_chart_respects_bar_limit():
    df = pd.DataFrame({"city": ["A", "A", "B", "C", "D"]})

    chart = viz.chart_data(df, "city", "bar", bar_limit=2)

    assert chart["bar_limit"] == 2
    assert chart["labels"] == ["A", "B"]


def test_bar_chart_defaults_to_all_categories():
    df = pd.DataFrame({"city": [f"City {i}" for i in range(20)]})

    chart = viz.chart_data(df, "city", "bar")

    assert chart["bar_limit"] is None
    assert len(chart["labels"]) == 20
