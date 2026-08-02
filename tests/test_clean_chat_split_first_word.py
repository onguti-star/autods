import pandas as pd

from backend.clean_chat import run_command


def test_split_first_word_from_named_column():
    df = pd.DataFrame({"full_name": ["Mr James", "Mrs Janes", "Dr Ada Lovelace"]})

    result_df, message = run_command(
        df,
        "split first word from full_name into title and name",
    )

    assert message == "Split first word from 'full_name' into 'title' and 'name'."
    assert result_df["title"].tolist() == ["Mr", "Mrs", "Dr"]
    assert result_df["name"].tolist() == ["James", "Janes", "Ada Lovelace"]


def test_separate_first_word_from_row_uses_only_text_column():
    df = pd.DataFrame({"customer": ["Mr James", "Mrs Janes", None, "Prince"]})

    result_df, _ = run_command(
        df,
        "separate the first word from the row into title and customer_name",
    )

    assert result_df["title"].tolist()[:2] == ["Mr", "Mrs"]
    assert result_df["customer_name"].tolist()[:2] == ["James", "Janes"]
    assert pd.isna(result_df.loc[2, "title"])
    assert pd.isna(result_df.loc[2, "customer_name"])
    assert result_df.loc[3, "title"] == "Prince"
    assert result_df.loc[3, "customer_name"] == ""

