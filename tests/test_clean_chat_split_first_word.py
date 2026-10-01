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


def test_split_into_two_columns_without_first_word_wording():
    # "split X into A and B" should work without needing the stiffer
    # "split first word from X into A and B" phrasing.
    df = pd.DataFrame({"sub_category": ["Mountain Bike", "Road Bike", "Mountain Tent"]})

    result_df, message = run_command(df, "split sub_category into mountain and bike")

    assert message == "Split first word from 'sub_category' into 'mountain' and 'bike'."
    assert result_df["mountain"].tolist() == ["Mountain", "Road", "Mountain"]
    assert result_df["bike"].tolist() == ["Bike", "Bike", "Tent"]
    # Source column is left untouched in the two-target form.
    assert result_df["sub_category"].tolist() == ["Mountain Bike", "Road Bike", "Mountain Tent"]


def test_split_into_single_column_leaves_remainder_in_source():
    # One target name given -> first word goes to the new column, and the
    # remainder overwrites the source column in place.
    df = pd.DataFrame({"sub_category": ["Mountain Bike", "Road Bike", "Mountain Tent", None]})

    result_df, message = run_command(df, "split sub_category into category_type")

    assert "new column 'category_type'" in message
    assert "stays in 'sub_category'" in message
    assert result_df["category_type"].tolist()[:3] == ["Mountain", "Road", "Mountain"]
    assert result_df["sub_category"].tolist()[:3] == ["Bike", "Bike", "Tent"]
    assert pd.isna(result_df.loc[3, "category_type"])
    assert pd.isna(result_df.loc[3, "sub_category"])


def test_split_first_word_from_phrasing_also_supports_single_target():
    df = pd.DataFrame({"sub_category": ["Mountain Bike", "Road Bike"]})

    result_df, message = run_command(
        df, "split first word from sub_category into category_type"
    )

    assert result_df["category_type"].tolist() == ["Mountain", "Road"]
    assert result_df["sub_category"].tolist() == ["Bike", "Bike"]
    assert message == (
        "Split first word from 'sub_category' into new column 'category_type'; "
        "the rest of the value now stays in 'sub_category'."
    )


def test_split_single_target_rejects_existing_column_name():
    df = pd.DataFrame({"sub_category": ["Mountain Bike"], "category_type": ["x"]})

    result_df, message = run_command(df, "split sub_category into category_type")

    assert "already exists" in message
    # Nothing should have changed.
    assert result_df["sub_category"].tolist() == ["Mountain Bike"]


def test_split_single_target_rejects_same_name_as_source():
    df = pd.DataFrame({"sub_category": ["Mountain Bike"]})

    _, message = run_command(df, "split sub_category into sub_category")

    # The new column name is literally the source column, which already
    # exists, so it's caught by the "already exists" check.
    assert "already exists" in message


def test_split_unrecognised_column_gives_available_columns():
    df = pd.DataFrame({"sub_category": ["Mountain Bike"]})

    _, message = run_command(df, "split wizzle into a and b")

    assert "couldn't find the source column" in message
    assert "sub_category" in message


def test_split_malformed_command_gives_targeted_suggestion_not_full_help():
    # The column is recognisable but the rest of the command isn't a
    # split/separate pattern we understand — should name the column and
    # show the exact phrasing, not dump the entire help text.
    df = pd.DataFrame({"sub_category": ["Mountain Bike"]})

    _, message = run_command(df, "split sub_category by bananas")

    assert "sub_category" in message
    assert "split sub_category into new_col_1" in message
    # Make sure it's the short targeted suggestion, not the full wall of
    # HELP_TEXT (which mentions many unrelated commands like "round" / "lowercase").
    assert "lowercase" not in message


def test_split_typo_in_verb_is_corrected():
    df = pd.DataFrame({"sub_category": ["Mountain Bike", "Road Bike"]})

    result_df, message = run_command(df, "seperate sub_category into cat1 and cat2")

    assert "I think you meant 'separate'" in message
    assert result_df["cat1"].tolist() == ["Mountain", "Road"]
    assert result_df["cat2"].tolist() == ["Bike", "Bike"]

