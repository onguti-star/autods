import numpy as np
import pandas as pd

from backend.hypothesis_testing import HypothesisTestRequest, run_hypothesis_test


def test_two_group_test_returns_inference_fields():
    df = pd.DataFrame({"sales": [10, 11, 12, 10, 11, 100, 101, 99, 102, 100],
                       "region": ["A"] * 5 + ["B"] * 5})
    result = run_hypothesis_test(df, HypothesisTestRequest(column_a="sales", column_b="region"))
    assert result["test"] == "Welch's two-sample t-test"
    assert result["p_value"] < 0.05
    assert "confidence_interval" in result
    assert "effect_size" in result
    assert result["significant"] is True


def test_correlation_and_chi_square_are_available():
    df = pd.DataFrame({"income": np.arange(20), "spend": np.arange(20) * 2,
                       "plan": ["basic", "pro"] * 10, "churn": ["yes", "no"] * 10})
    corr = run_hypothesis_test(df, HypothesisTestRequest(column_a="income", column_b="spend"))
    assert corr["test"] == "Pearson correlation test"
    chi = run_hypothesis_test(df, HypothesisTestRequest(column_a="plan", column_b="churn"))
    assert chi["test"] == "Chi-square test of independence"
    assert chi["effect_size_name"] == "Cramer's V"
