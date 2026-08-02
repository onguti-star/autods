import pandas as pd

from backend.clean import run_command as run_legacy_command
from backend.clean_chat import run_command


def test_rename_accepts_into_as_to_synonym():
    df = pd.DataFrame({"time": ["09:00"], "amount": [10]})

    result_df, message = run_command(df, "rename time into invoice_time")

    assert message == "Renamed column 'time' to 'invoice_time'."
    assert "invoice_time" in result_df.columns
    assert "time" not in result_df.columns


def test_rename_accepts_as_and_optional_column_word():
    df = pd.DataFrame({"Time": ["09:00"], "amount": [10]})

    result_df, message = run_command(df, "rename column Time as column invoice_time")

    assert message == "Renamed column 'Time' to 'invoice_time'."
    assert "invoice_time" in result_df.columns
    assert "Time" not in result_df.columns


def test_drop_column_accepts_remove_delete_discard_and_erase_synonyms():
    df = pd.DataFrame({"notes": ["x"], "amount": [10]})

    for command in (
        "remove column notes",
        "delete column notes",
        "discard column notes",
        "erase column notes",
        "get rid of column notes",
    ):
        result_df, message = run_command(df, command)
        assert message == "Dropped column 'notes'."
        assert list(result_df.columns) == ["amount"]


def test_discard_rows_where_value_matches_remove_rows():
    df = pd.DataFrame({"status": ["paid", "cancelled", "paid"]})

    result_df, message = run_command(df, "discard rows where status is cancelled")

    assert message == "Removed 1 row(s) where 'status' was 'cancelled'."
    assert result_df["status"].tolist() == ["paid", "paid"]


def test_legacy_clean_parser_accepts_rename_into():
    df = pd.DataFrame({"time": ["09:00"], "amount": [10]})

    result_df, message = run_legacy_command(df, "rename time into invoice_time")

    assert message == "Renamed column 'time' to 'invoice_time'."
    assert "invoice_time" in result_df.columns
