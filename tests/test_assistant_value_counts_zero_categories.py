import pandas as pd

from backend.assistant import _value_counts_table


def test_value_counts_omits_unused_categories_after_filtering():
    country = pd.Series(
        pd.Categorical(
            ["United States", "United States", "India", "India"],
            categories=["United States", "India", "Kenya", "France"],
        ),
        name="country",
    )
    df = pd.DataFrame({"country": country})

    table = _value_counts_table(df, "country")

    assert [row["country"] for row in table["rows"]] == ["United States", "India"]
    assert all(row["count"] > 0 for row in table["rows"])
