import pandas as pd

from backend.assistant import answer_question, answer_question_table


def _loan_df():
    return pd.DataFrame(
        {
            "loan_status": [
                "Fully Paid",
                "Charged Off",
                "Fully Paid",
                "Charged Off",
                "Current",
                "Fully Paid",
            ],
            "grade": ["A", "A", "B", "B", "B", "C"],
            "purpose": ["car", "car", "debt", "debt", "debt", "home"],
            "int_rate": [6.0, 9.0, 12.0, 14.0, 15.0, 18.0],
            "loan_amnt": [1000, 1200, 2000, 2200, 2400, 3000],
        }
    )


def test_distinct_value_counts_question_returns_counts_table():
    df = _loan_df()

    answer = answer_question(df, "check the distinct loan_status values and their counts")
    table = answer_question_table(df, "check the distinct loan_status values and their counts")

    assert "Distinct values in 'loan_status'" in answer
    assert table["columns"] == ["loan_status", "count", "pct"]
    assert table["rows"][0]["loan_status"] == "Fully Paid"
    assert table["rows"][0]["count"] == 3


def test_default_rate_by_grade_matches_database_style_grouping():
    df = _loan_df()

    answer = answer_question(df, "what is the default rate by grade?")
    table = answer_question_table(df, "what is the default rate by grade?")

    assert "Charged Off Rate Pct by 'grade'" in answer
    assert table["columns"] == ["grade", "total_rows", "Charged Off_count", "Charged Off_rate_pct"]
    rows = {row["grade"]: row for row in table["rows"]}
    assert rows["A"]["total_rows"] == 2
    assert rows["A"]["Charged Off_rate_pct"] == 50.0
    assert rows["B"]["total_rows"] == 2
    assert rows["B"]["Charged Off_rate_pct"] == 50.0


def test_default_rate_excludes_current_loans_when_fully_paid_exists():
    df = _loan_df()

    table = answer_question_table(df, "show default rate by purpose")
    rows = {row["purpose"]: row for row in table["rows"]}

    assert rows["debt"]["total_rows"] == 2
    assert rows["debt"]["Charged Off_count"] == 1
    assert rows["debt"]["Charged Off_rate_pct"] == 50.0


def test_multiple_numeric_averages_by_group_return_one_table():
    df = _loan_df()

    answer = answer_question(df, "average int_rate and loan_amnt by grade")
    table = answer_question_table(df, "average int_rate and loan_amnt by grade")

    assert "Grouped numeric summary by 'grade'" in answer
    assert table["columns"] == ["grade", "rows", "mean_int_rate", "mean_loan_amnt"]
    rows = {row["grade"]: row for row in table["rows"]}
    assert rows["A"]["mean_int_rate"] == 7.5
    assert rows["B"]["mean_loan_amnt"] == 2200.0
