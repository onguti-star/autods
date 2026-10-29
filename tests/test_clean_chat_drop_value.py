import pandas as pd

from backend.clean_chat import run_command


def test_drop_exact_numeric_value_in_column():
    df = pd.DataFrame({"age": [28.0, 0.8299999833106995, 31.0], "name": ["a", "b", "c"]})

    result_df, message = run_command(df, "drop 0.8299999833106995 in age")

    assert message == "Removed 1 row(s) where 'age' was 0.8299999833106995."
    assert result_df["name"].tolist() == ["a", "c"]


def test_drop_value_accepts_delete_and_text_values():
    df = pd.DataFrame({"gender": ["male", "female", "male"], "score": [1, 2, 3]})

    result_df, message = run_command(df, "delete female from gender")

    assert message == "Removed 1 row(s) where 'gender' was female."
    assert result_df["gender"].tolist() == ["male", "male"]


def test_unknown_removal_command_gets_short_relevant_suggestion():
    df = pd.DataFrame({"age": [20, 30]})

    result_df, message = run_command(df, "drop this weird thing age")

    assert result_df.equals(df)
    assert "drop VALUE in age" in message
    assert "I understand plain-English" not in message


def test_remove_punctuation_from_column_is_not_treated_as_drop_rows():
    df = pd.DataFrame({"first_name": ["Harris, Owen", "Bradley, John"]})

    result_df, message = run_command(df, "remove , in first name")

    assert "meant 'named'" not in message
    assert result_df["first_name"].tolist() == ["Harris Owen", "Bradley John"]
