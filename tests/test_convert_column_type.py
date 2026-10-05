import pandas as pd

from backend.main import _convert_column_type


def test_mixed_separator_dates_both_parse_to_same_format():
    # Regression test: pandas' to_datetime infers ONE format from the first
    # value in a column and applies it to the whole series unless told
    # otherwise. A column mixing "01-09-2026" (dash) and "09/09/2026"
    # (slash) used to silently turn whichever style didn't match the
    # guessed format into missing data (NaT) instead of erroring — a real
    # user-reported bug. format="mixed" fixes this by inferring per-row.
    df = pd.DataFrame({"order_date": ["01-09-2026", "09/09/2026", "15-09-2026"]})

    out, message = _convert_column_type(df, "order_date", "date")

    assert out["order_date"].tolist() == ["2026-09-01", "2026-09-09", "2026-09-15"]
    assert "could not be converted" not in message


def test_genuinely_invalid_date_values_still_become_missing():
    df = pd.DataFrame({"order_date": ["01-09-2026", "not a date", ""]})

    out, message = _convert_column_type(df, "order_date", "date")

    assert out["order_date"].tolist()[0] == "2026-09-01"
    assert pd.isna(out["order_date"].tolist()[1])
    assert pd.isna(out["order_date"].tolist()[2])
    assert "2 value(s) could not be converted" in message


def test_mixed_separator_dates_also_work_for_full_datetime_conversion():
    df = pd.DataFrame({"order_date": ["01-09-2026", "09/09/2026"]})

    out, _ = _convert_column_type(df, "order_date", "datetime")

    assert out["order_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-09-01", "2026-09-09"]
