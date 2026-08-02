import pandas as pd

from backend.store import Session


def test_clear_current_training_keeps_saved_runs():
    session = Session(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "data.csv")
    session.target = "y"
    session.problem_type = "regression"
    session.models = {"model": object()}
    session.leaderboard = [{"model": "model"}]
    session.best_model_name = "model"
    session.feature_columns = ["x"]
    session.label_encoder = object()
    session.progress_messages = ["Training complete!"]
    session.saved_runs["run_1"] = {"best_model_name": "model"}

    session.clear_current_training()

    assert session.target is None
    assert session.problem_type is None
    assert session.models == {}
    assert session.leaderboard == []
    assert session.best_model_name is None
    assert session.feature_columns == []
    assert session.label_encoder is None
    assert session.progress_messages == []
    assert session.saved_runs == {"run_1": {"best_model_name": "model"}}
