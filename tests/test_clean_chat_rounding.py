import pandas as pd

from backend.clean_chat import run_command


def test_round_column_to_2_decimal_places():
    df = pd.DataFrame({"price": [1.12345678, 2.98765432, 3.50000001]})

    result_df, message = run_command(df, "round price to 2 decimal places")

    assert message == "Rounded 'price' to 2 decimal places."
    assert result_df["price"].tolist() == [1.12, 2.99, 3.5]


def test_round_column_to_2_dp():
    df = pd.DataFrame({"latitude": [12.34567890, -45.98765432]})

    result_df, message = run_command(df, "round latitude to 2 dp")

    assert message == "Rounded 'latitude' to 2 decimal places."
    assert result_df["latitude"].tolist() == [12.35, -45.99]


def test_round_column_to_0_decimals():
    df = pd.DataFrame({"count": [1.4, 2.6, 3.5]})

    result_df, message = run_command(df, "round count to 0 decimal places")

    assert message == "Rounded 'count' to whole number."
    assert result_df["count"].tolist() == [1, 3, 4]


def test_round_column_to_3_decimals():
    df = pd.DataFrame({"value": [1.123456789, 2.987654321]})

    result_df, message = run_command(df, "round value to 3 decimals")

    assert message == "Rounded 'value' to 3 decimal places."
    assert result_df["value"].tolist() == [1.123, 2.988]


def test_keep_n_decimals_in_column():
    df = pd.DataFrame({"longitude": [100.12345678, 200.98765432]})

    result_df, message = run_command(df, "keep 2 decimals in longitude")

    assert message == "Rounded 'longitude' to 2 decimal places."
    assert result_df["longitude"].tolist() == [100.12, 200.99]


def test_round_all_numeric_columns():
    df = pd.DataFrame({
        "price": [1.12345678, 2.98765432],
        "tax": [0.11111111, 0.22222222],
        "name": ["a", "b"],
    })

    result_df, message = run_command(df, "round all numeric columns to 2 decimal places")

    assert "price" in message
    assert "tax" in message
    assert result_df["price"].tolist() == [1.12, 2.99]
    assert result_df["tax"].tolist() == [0.11, 0.22]


def test_round_nonexistent_column():
    df = pd.DataFrame({"price": [1.12345678]})

    result_df, message = run_command(df, "round nonexistent to 2 decimal places")

    assert "couldn't find" in message.lower()
    assert result_df["price"].tolist() == [1.12345678]


def test_round_non_numeric_column():
    df = pd.DataFrame({"name": ["Alice", "Bob"]})

    result_df, message = run_command(df, "round name to 2 decimal places")

    assert "isn't numeric" in message.lower()
    assert result_df["name"].tolist() == ["Alice", "Bob"]


def test_round_preserves_missing_values():
    df = pd.DataFrame({"price": [1.12345678, None, 3.50000001]})

    result_df, message = run_command(df, "round price to 2 decimal places")

    assert message == "Rounded 'price' to 2 decimal places."
    assert result_df["price"].iloc[0] == 1.12
    assert pd.isna(result_df["price"].iloc[1])
    assert result_df["price"].iloc[2] == 3.5


def test_round_too_many_decimals():
    df = pd.DataFrame({"price": [1.12345678]})

    result_df, message = run_command(df, "round price to 20 decimal places")

    assert "between 0 and 15" in message.lower()
    assert result_df["price"].tolist() == [1.12345678]
