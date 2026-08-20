import pandas as pd

from backend.clean import run_command as run_legacy_command
from backend.clean_chat import run_command


def test_create_empty_column_then_fill_with_expression():
    df = pd.DataFrame({"price": [10, 20, 30], "quantity": [2, 3, 4]})

    result_df, message = run_command(df, "create a new column called revenue")

    assert message == "Created empty column 'revenue'."
    assert "revenue" in result_df.columns
    assert result_df["revenue"].isna().all()

    filled_df, fill_message = run_command(result_df, "fill revenue with price * quantity")

    assert fill_message == "Filled column 'revenue' with expression: price * quantity"
    assert filled_df["revenue"].tolist() == [20, 60, 120]


def test_fill_it_uses_only_empty_column():
    df = pd.DataFrame({"price": [5, 8], "quantity": [4, 2]})
    result_df, _ = run_command(df, "create column revenue")

    filled_df, message = run_command(result_df, "fill it with price*quantity")

    assert message == "Filled column 'revenue' with expression: price*quantity"
    assert filled_df["revenue"].tolist() == [20, 16]


def test_update_and_set_are_fill_expression_synonyms():
    df = pd.DataFrame({
        "price": [100, 50],
        "quantity": [3, 6],
        "revenue": [pd.NA, pd.NA],
    })

    updated_df, update_message = run_command(df, "update revenue with price * quantity")
    set_df, set_message = run_command(df, "set revenue to price * quantity")

    assert update_message == "Filled column 'revenue' with expression: price * quantity"
    assert set_message == "Filled column 'revenue' with expression: price * quantity"
    assert updated_df["revenue"].tolist() == [300, 300]
    assert set_df["revenue"].tolist() == [300, 300]


def test_fill_missing_values_still_uses_missing_value_handler():
    df = pd.DataFrame({"income": [10, None, 30]})

    result_df, message = run_command(df, "fill missing values in income with median")

    assert message == "Filled 1 missing value(s) in 'income' with 20."
    assert result_df["income"].tolist() == [10, 20, 30]


def test_legacy_parser_create_empty_column_and_fill_expression():
    df = pd.DataFrame({"price": [10, 20], "quantity": [2, 3]})

    result_df, _ = run_legacy_command(df, "create a new column called revenue")
    filled_df, message = run_legacy_command(result_df, "fill revenue with price * quantity")

    assert message == "Filled column 'revenue' with expression: price * quantity"
    assert filled_df["revenue"].tolist() == [20, 60]
