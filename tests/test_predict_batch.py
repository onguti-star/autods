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


# ---- batch predict is kept with BOTH datasets and survives into exports ----
import base64
import json
import re

import pytest
from fastapi import HTTPException

from backend import nb
from backend.main_report import _build_html_report


def _run_batch():
    model_session = _make_trained_session()
    target_session = _make_target_session()
    result = main.predict_batch(model_session.id, main.BatchPredictRequest(target_session_id=target_session.id))
    return model_session, target_session, result


def test_batch_record_is_stored_on_both_datasets():
    model_session, target_session, result = _run_batch()
    pid = result["saved_prediction"]["id"]
    assert model_session.saved_predictions[pid].get("role") is None
    scored = target_session.saved_predictions[pid]
    assert scored["role"] == "scored"
    assert scored["saved_column"] == "predicted_Survived"
    assert scored["model_dataset"] == "train.csv"


def test_html_report_of_scored_dataset_explains_the_prediction_column():
    _, target_session, _ = _run_batch()
    html = _build_html_report(target_session)
    assert 'id="predictions"' in html
    assert "Batch predictions added to this dataset" in html
    assert "predicted_Survived" in html and "train.csv" in html


def test_html_report_of_model_dataset_describes_the_batch_not_one_row():
    model_session, _, _ = _run_batch()
    html = _build_html_report(model_session)
    assert "Batch prediction: Survived (3 rows scored)" in html
    assert "scored_dataset:" not in html          # the old one-row "Inputs:" dump


@pytest.mark.parametrize("which", ["model", "scored"])
def test_notebook_has_batch_section_and_runs(which):
    model_session, target_session, _ = _run_batch()
    session = model_session if which == "model" else target_session
    notebook = json.loads(nb.build_notebook(session))
    namespace = {}
    text = ""
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        text += source + "\n"
        if cell["cell_type"] == "code":
            exec(source, namespace)               # every code cell must run
    assert "Batch prediction" in text
    if which == "scored":
        assert "predicted_Survived" in namespace["df"].columns


def test_batch_csv_endpoint_rebuilds_full_and_submission_files():
    model_session, _, result = _run_batch()
    pid = result["saved_prediction"]["id"]
    full = main.predict_batch_csv(model_session.id, pid, "full").body.decode()
    assert full.splitlines()[0] == "PassengerId,Age,Fare,Survived"
    assert len(full.strip().splitlines()) == 4
    sub = main.predict_batch_csv(model_session.id, pid, "submission").body.decode()
    assert sub.splitlines()[0] == "PassengerId,Survived"


def test_batch_csv_endpoint_reports_a_removed_column_clearly():
    model_session, target_session, result = _run_batch()
    pid = result["saved_prediction"]["id"]
    target_session.df = target_session.df.drop(columns=["predicted_Survived"])
    with pytest.raises(HTTPException) as err:
        main.predict_batch_csv(model_session.id, pid, "full")
    assert err.value.status_code == 404 and "Run Batch predict again" in err.value.detail
