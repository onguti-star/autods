import pandas as pd

from backend.clean_chat import run_command


def test_extract_title_from_kaggle_style_name():
    df = pd.DataFrame({
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley",
            "Heikkinen, Miss. Laina",
        ],
    })

    result_df, message = run_command(df, "extract title from Name into Title")

    assert "Extracted title from 'Name' into 'Title'" in message
    assert result_df["Title"].tolist() == ["Mr", "Mrs", "Miss"]


def test_extract_deck_from_cabin_code():
    df = pd.DataFrame({"Cabin": ["C85", None, "E46", "B96 B98"]})

    result_df, message = run_command(df, "extract deck from Cabin into Deck")

    assert "3 of 4 row(s) matched" in message
    assert result_df["Deck"].tolist()[0] == "C"
    assert pd.isna(result_df["Deck"].tolist()[1])
    assert result_df["Deck"].tolist()[2] == "E"
    assert result_df["Deck"].tolist()[3] == "B"


def test_extract_first_letter_generic():
    df = pd.DataFrame({"code": ["A1", "b2", None]})

    result_df, _ = run_command(df, "extract first letter from code into initial")

    assert result_df["initial"].tolist()[0] == "A"
    assert result_df["initial"].tolist()[1] == "b"
    assert pd.isna(result_df["initial"].tolist()[2])


def test_extract_custom_regex_pattern():
    df = pd.DataFrame({"sku": ["SKU-1234-A", "SKU-5678-B"]})

    result_df, message = run_command(
        df, "extract pattern SKU-(\\d+) from sku into sku_number"
    )

    assert "Extracted pattern from 'sku' into 'sku_number'" in message
    assert result_df["sku_number"].tolist() == ["1234", "5678"]


def test_extract_rejects_existing_column_name():
    df = pd.DataFrame({"Name": ["Braund, Mr. Owen Harris"], "Title": ["x"]})

    result_df, message = run_command(df, "extract title from Name into Title")

    assert "already exists" in message
    assert result_df.equals(df)


def test_extract_reports_missing_source_column():
    df = pd.DataFrame({"Name": ["Braund, Mr. Owen Harris"]})

    _, message = run_command(df, "extract title from Passenger into Title")

    assert "couldn't find the column" in message


def test_extract_rejects_invalid_regex():
    df = pd.DataFrame({"Cabin": ["C85"]})

    result_df, message = run_command(df, "extract pattern ( from Cabin into Deck")

    assert "isn't a valid pattern" in message
    assert result_df.equals(df)


def test_extract_source_column_name_is_not_typo_corrected():
    # "Name" directly follows "from" and is close in edit-distance to the
    # keyword "named" — this must resolve to the column, not get "corrected"
    # into an unrelated keyword and produce a confusing typo-correction note.
    df = pd.DataFrame({"Name": ["Braund, Mr. Owen Harris"]})

    _, message = run_command(df, "extract title from Name into Title")

    assert "I think you meant" not in message
