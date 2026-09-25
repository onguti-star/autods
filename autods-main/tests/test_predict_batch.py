import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from backend import main, store


def _make_trained_session():
    """A session that looks like a finished 'train.csv' training run: two
    numeric features (Age, Fare) predicting Survived, wrapped exactly the way
    automl.py wraps a real model (a fitted sklearn Pipeline with an imputer,
    since a real trained pipeline always tolerates missing feature values)."""
    df = pd.DataFrame({
        "PassengerId": [1, 2, 3, 4],
        "Age": [22, 38, 26, 35],
        "Fare": [7.25, 71.28, 7.92, 53.1],
        "Survived": [0, 1, 1, 1],
    })
    session = store.create_session(df, "train.csv")

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Survived"])
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression()),
    ])
    pipe.fit(df[["Age", "Fare"]], y)

    session.target = "Survived"
    session.problem_type = "classification"
    session.models = {"LogisticRegression": pipe}
    session.best_model_name = "LogisticRegression"
    session.feature_columns = ["Age", "Fare"]
    session.label_encoder = encoder
    return session


def _make_target_session():
    """A 'test.csv'-style session: same feature columns, no target column,
    plus a PassengerId column that should get auto-detected as the id column."""
    df = pd.DataFrame({
        "PassengerId": [892, 893, 894],
        "Age": [34.5, 47, 62],
        "Fare": [7.83, 7.0, 9.69],
    })
    return store.create_session(df, "test.csv")


def test_batch_predict_scores_every_row_of_a_different_dataset():
    model_session = _make_trained_session()
    target_session = _make_target_session()

    req = main.BatchPredictRequest(target_session_id=target_session.id)
    result = main.predict_batch(model_session.id, req)

    assert result["ok"] is True
    assert result["target"] == "Survived"
    assert result["rows_predicted"] == 3
    assert result["missing_feature_columns"] == []
    assert result["id_column"] == "PassengerId"

    full_lines = result["full_csv"].strip().splitlines()
    assert full_lines[0] == "PassengerId,Age,Fare,Survived"
    assert len(full_lines) == 4  # header + 3 rows

    submission_lines = result["submission_csv"].strip().splitlines()
    assert submission_lines[0] == "PassengerId,Survived"
    assert len(submission_lines) == 4

    assert len(result["preview"]) == 3
    assert result["preview"][0]["PassengerId"] == 892


def test_batch_predict_reports_missing_feature_columns_but_still_runs():
    model_session = _make_trained_session()
    df = pd.DataFrame({"PassengerId": [1, 2], "Age": [40, 50]})  # no Fare
    target_session = store.create_session(df, "partial.csv")

    req = main.BatchPredictRequest(target_session_id=target_session.id)
    result = main.predict_batch(model_session.id, req)

    assert result["missing_feature_columns"] == ["Fare"]
    assert result["rows_predicted"] == 2


def test_batch_predict_uses_a_saved_run_when_given_a_run_id():
    model_session = _make_trained_session()
    # Snapshot the current training as a saved run, then clear the live model
    # so the only way this can succeed is by actually using the saved run.
    model_session.saved_runs["run_1"] = {
        "name": "Auto-saved: Survived",
        "target": "Survived",
        "problem_type": "classification",
        "models": model_session.models,
        "label_encoder": model_session.label_encoder,
        "feature_columns": model_session.feature_columns,
        "best_model_name": model_session.best_model_name,
    }
    model_session.models = {}

    target_session = _make_target_session()
    req = main.BatchPredictRequest(target_session_id=target_session.id, run_id="run_1")
    result = main.predict_batch(model_session.id, req)

    assert result["source_name"] == "Auto-saved: Survived"
    assert result["rows_predicted"] == 3


def test_batch_predict_rejects_empty_target_dataset():
    import pytest
    from fastapi import HTTPException

    model_session = _make_trained_session()
    target_session = store.create_session(pd.DataFrame({"PassengerId": [], "Age": [], "Fare": []}), "empty.csv")

    req = main.BatchPredictRequest(target_session_id=target_session.id)
    with pytest.raises(HTTPException) as exc_info:
        main.predict_batch(model_session.id, req)
    assert exc_info.value.status_code == 400
