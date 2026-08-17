import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend import nb
from backend.store import Session, delete_session


def _notebook_text(notebook_json: str) -> str:
    notebook = json.loads(notebook_json)
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_notebook_export_only_includes_completed_work():
    session = Session(pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]}), "data.csv")
    try:
        session.cleaning_log = ["Removed 1 duplicate row"]
        session.target = "y"
        session.problem_type = "regression"
        session.best_model_name = "Random Forest"
        session.feature_columns = ["x"]
        session.leaderboard = [
            {"model": "Random Forest", "metrics": {"r2": 0.9}, "primary_score": 0.9}
        ]

        text = _notebook_text(nb.build_notebook(session))

        assert "Completed Work Notebook" in text
        assert "Cleaning Done" in text
        assert "Model Training Done" in text
        assert "Removed 1 duplicate row" in text
        assert "target: y" not in text  # target is stored as data, not hard-coded output text
        assert "Principal Component Analysis" not in text
        assert "Ask Questions About Your Data" not in text
        assert "def build_preprocessor(" not in text
        assert "def answer_question(" not in text
    finally:
        delete_session(session.id)


def test_notebook_export_has_simple_data_fallback_when_no_work_recorded():
    session = Session(pd.DataFrame({"x": [1, 2], "name": ["a", "b"]}), "data.csv")
    try:
        text = _notebook_text(nb.build_notebook(session))

        assert "Data Notebook" in text
        assert "No cleaning, training, prediction, visualization, or unsupervised actions were recorded yet." in text
        assert "df.describe(include='all')" in text
    finally:
        delete_session(session.id)
