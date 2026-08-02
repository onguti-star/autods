import pandas as pd

from backend.assistant import answer_question


def test_plain_mean_question_returns_overall_mean_not_grouped_mean():
    df = pd.DataFrame(
        {
            "consume": [4.0, 5.0, 6.0],
            "gas_type": ["E10", "SP98", "E10"],
        }
    )

    answer = answer_question(df, "what is the mean of consume")

    assert answer == "The average 'consume' is 5.00."


def test_explicit_group_mean_question_still_groups():
    df = pd.DataFrame(
        {
            "consume": [4.0, 5.0, 6.0],
            "gas_type": ["E10", "SP98", "E10"],
        }
    )

    answer = answer_question(df, "what is the mean of consume by gas_type")

    assert answer.startswith("Average 'consume' by 'gas_type'")
