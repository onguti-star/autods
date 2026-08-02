import pandas as pd

from backend.assistant import answer_question, answer_question_table, execute_action
from backend.clean_chat import run_command


def test_main_assistant_corrects_misspelled_correlation_keyword():
    df = pd.DataFrame(
        {
            "sales": [1, 2, 3, 4],
            "revenue": [10, 20, 30, 40],
            "cost": [8, 15, 22, 30],
        }
    )

    answer = answer_question(df, "show corrrelation for sales")
    table = answer_question_table(df, "show corrrelation for sales")

    assert "I think you meant 'correlation' instead of 'corrrelation'." in answer
    assert "sales vs revenue" in answer
    assert table is not None
    assert table["columns"] == ["Correlation", "Value", "Strength"]


def test_main_assistant_action_corrects_misspelled_remove_duplicates():
    df = pd.DataFrame({"name": ["A", "A", "B"], "value": [1, 1, 2]})

    result = execute_action(df, "remvoe duplicats")

    assert result["success"] is True
    assert len(result["modified_df"]) == 2
    assert "'remove' instead of 'remvoe'" in result["message"]
    assert "'duplicates' instead of 'duplicats'" in result["message"]


def test_clean_assistant_corrects_misspelled_remove_duplicates():
    df = pd.DataFrame({"name": ["A", "A", "B"], "value": [1, 1, 2]})

    result_df, message = run_command(df, "remvoe duplicats")

    assert len(result_df) == 2
    assert "'remove' instead of 'remvoe'" in message
    assert "'duplicates' instead of 'duplicats'" in message

