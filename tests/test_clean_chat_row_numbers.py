import pandas as pd

from backend.clean import run_command as run_legacy_command
from backend.clean_chat import run_command


def _df() -> pd.DataFrame:
    return pd.DataFrame({"name": list("abcdef"), "amount": [10, 20, 30, 40, 50, 60]})


def test_remove_single_row_by_number_is_one_based():
    df = _df()

    result_df, message = run_command(df, "remove row 2")

    assert message == "Removed row 2."
    assert result_df["name"].tolist() == ["a", "c", "d", "e", "f"]
    # The input dataframe must never be mutated in place.
    assert df["name"].tolist() == list("abcdef")


def test_remove_row_accepts_number_and_hash_variants():
    for command in ("delete row number 4", "drop row #4", "discard row no. 4", "get rid of row 4"):
        result_df, message = run_command(_df(), command)
        assert message == "Removed row 4."
        assert result_df["name"].tolist() == ["a", "b", "c", "e", "f"]


def test_remove_multiple_rows_comma_and_and_list():
    result_df, message = run_command(_df(), "remove rows 2, 4 and 6")

    assert message == "Removed rows 2, 4, 6."
    assert result_df["name"].tolist() == ["a", "c", "e"]


def test_remove_row_range_with_to_and_dash():
    result_df, message = run_command(_df(), "remove rows 2 to 4")

    assert message == "Removed rows 2, 3, 4."
    assert result_df["name"].tolist() == ["a", "e", "f"]

    result_df, message = run_command(_df(), "drop rows 5-6")

    assert message == "Removed rows 5, 6."
    assert result_df["name"].tolist() == ["a", "b", "c", "d"]


def test_remove_row_accepts_trailing_from_the_dataset():
    result_df, message = run_command(_df(), "remove row 3 from the dataset")

    assert message == "Removed row 3."
    assert result_df["name"].tolist() == ["a", "b", "d", "e", "f"]


def test_remove_row_out_of_range_makes_no_change():
    df = _df()

    result_df, message = run_command(df, "remove row 99")

    assert result_df["name"].tolist() == list("abcdef")
    assert "out of range" in message
    assert "6 row(s)" in message
    assert "Nothing removed" in message


def test_remove_row_zero_is_rejected():
    result_df, message = run_command(_df(), "remove row 0")

    assert result_df["name"].tolist() == list("abcdef")
    assert "out of range" in message


def test_duplicate_row_numbers_remove_the_row_once():
    result_df, message = run_command(_df(), "remove rows 3, 3")

    assert message == "Removed row 3."
    assert result_df["name"].tolist() == ["a", "b", "d", "e", "f"]


def test_row_number_command_does_not_break_condition_filtering():
    result_df, message = run_command(_df(), "remove rows where amount is 30")

    assert message == "Removed 1 row(s) where 'amount' was '30'."
    assert result_df["name"].tolist() == ["a", "b", "d", "e", "f"]


def test_legacy_clean_parser_also_removes_rows_by_number():
    result_df, message = run_legacy_command(_df(), "delete row number 2")

    assert message == "Removed row 2."
    assert result_df["name"].tolist() == ["a", "c", "d", "e", "f"]
