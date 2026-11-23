import pandas as pd

from backend.clean_chat import run_command


def test_polite_equivalent_removal_wording():
    df = pd.DataFrame({"status": ["ok", None, "ok"]})
    result, message = run_command(df, "Could you please get rid of rows with missing values in status")
    assert len(result) == 2
    assert "Removed 1" in message


def test_round_accepts_number_before_column_word_order():
    df = pd.DataFrame({"price": [1.234, 2.345]})
    result, message = run_command(df, "Please round to 1 decimal place in price")
    assert result["price"].tolist() == [1.2, 2.3]
    assert "1 decimal place" in message


def test_sort_accepts_direction_before_column():
    df = pd.DataFrame({"age": [3, 1, 2]})
    result, message = run_command(df, "descending sort by age")
    assert result["age"].tolist() == [3, 2, 1]
    assert "descending" in message
