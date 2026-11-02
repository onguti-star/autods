import pandas as pd

from backend.clean_chat import run_command


def test_remove_rows_with_low_value_frequency_in_column():
    df = pd.DataFrame({"rating": [1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5]})

    result, message = run_command(
        df,
        "remove rows where value count is less than 3 in rating",
    )

    assert result["rating"].tolist() == [3, 3, 3, 4, 4, 4, 4]
    assert "Removed 4 row(s)" in message


def test_frequency_filter_is_not_treated_as_read_only_analysis():
    df = pd.DataFrame({"rating": [1, 1, 2, 3, 3]})

    result, message = run_command(df, "remove rows where rating value count is fewer than 2")

    assert result["rating"].tolist() == [1, 1, 3, 3]
    assert "Distinct values" not in message
