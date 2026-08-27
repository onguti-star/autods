import pandas as pd

from backend.clean_chat import run_command, run_command_with_table


def _loan_df():
    return pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Charged Off", "Current"],
            "grade": ["A", "A", "B", "B", "B"],
            "int_rate": [6.0, 9.0, 12.0, 14.0, 15.0],
            "loan_amnt": [1000, 1200, 2000, 2200, 2400],
        }
    )


def test_clean_chat_can_show_distinct_counts_without_changing_data():
    df = _loan_df()

    result_df, message, table = run_command_with_table(
        df,
        "check distinct loan_status values and counts",
    )

    assert result_df is df
    assert "Distinct values in 'loan_status'" in message
    assert table["columns"] == ["loan_status", "count", "pct"]


def test_clean_chat_can_show_default_rate_table_without_changing_data():
    df = _loan_df()

    result_df, message, table = run_command_with_table(df, "show default rate by grade")

    assert result_df is df
    assert "Charged Off Rate Pct by 'grade'" in message
    assert table["columns"] == ["grade", "total_rows", "Charged Off_count", "Charged Off_rate_pct"]


def test_clean_chat_two_value_api_stays_backwards_compatible():
    df = _loan_df()

    result_df, message = run_command(df, "show default rate by grade")

    assert result_df is df
    assert "Charged Off Rate Pct by 'grade'" in message
