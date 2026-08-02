import pandas as pd

from backend.eda import learning_recommendation


def test_churned_column_is_recommended_as_classification_target():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "monthly_spend": [30, 45, 20, 75, 50, 35],
            "support_calls": [0, 1, 4, 0, 2, 3],
            "churned": ["no", "no", "yes", "no", "yes", "yes"],
        }
    )

    rec = learning_recommendation(df)

    assert rec["mode"] == "supervised"
    assert rec["suggested_targets"][0]["column"] == "churned"
    assert rec["suggested_targets"][0]["problem_type"] == "classification"

