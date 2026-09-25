import pandas as pd

from backend.clean import build_cleaning_recommendations


def test_build_cleaning_recommendations_flags_common_actions():
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Alice", "Bob"],
            "Age": [30, None, 40],
            "City": ["NY", None, "LA"],
            "Flag": ["x", "x", "x"],
            "Amount": ["10", "20", "30"],
        }
    )

    recommendations = build_cleaning_recommendations(df)

    assert recommendations["summary"].startswith("I found")
    assert any(issue["key"] == "duplicates" for issue in recommendations["issues"])
    assert any(issue["key"] == "constant_columns" for issue in recommendations["issues"])
    assert any(issue["key"] == "mixed_types" for issue in recommendations["issues"])
    assert recommendations["recommended_options"]["drop_duplicates"] is True
    assert recommendations["recommended_options"]["drop_constant_cols"] is True
    assert recommendations["recommended_options"]["fix_mixed_types"] is True
    assert recommendations["recommended_options"]["fill_missing_numeric"] == "median"
    assert recommendations["recommended_options"]["fill_missing_categorical"] == "mode"
