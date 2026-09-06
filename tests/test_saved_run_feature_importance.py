"""
Regression test: saved/auto-saved training runs used to drop feature
importance entirely, so a model card built from a saved run (as opposed to
the "live" just-trained model) had no bars to show. Both the manual
/api/train_runs save and the automatic "auto-saved" snapshot taken when
training a new target should carry feature_importance through so every
model card -- live or saved -- can render its own chart.
"""
import numpy as np
import pandas as pd

from backend import automl, main, store


def _make_session():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "target": rng.integers(0, 2, size=n),
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
        }
    )
    return store.create_session(df, "test.csv")


def _train_into_session(session, target="target"):
    problem_type, leaderboard, fitted, best_name, label_encoder = automl.train_all(
        session.df, target, progress_callback=lambda _: None
    )
    session.target = target
    session.problem_type = problem_type
    session.models = fitted
    session.leaderboard = leaderboard
    session.best_model_name = best_name
    session.label_encoder = label_encoder
    session.feature_columns = [c for c in session.df.columns if c != target]


def test_manual_save_run_includes_feature_importance():
    session = _make_session()
    _train_into_session(session)

    resp = main.save_train_run(session.id, main.SaveRunRequest(name="My run"))

    assert resp["saved"]["feature_importance"], "expected non-empty feature importance on saved run"
    assert resp["runs"][0]["feature_importance"] == resp["saved"]["feature_importance"]


def test_auto_saved_run_when_switching_target_includes_feature_importance():
    session = _make_session()
    session.df["target2"] = (session.df["num1"] > 0).astype(int)
    _train_into_session(session, target="target")

    # Simulate what /api/train/{id} does when switching to a new target: it
    # auto-saves the previous target's model before starting a fresh run.
    run_id = "fake-run-id"
    session.saved_runs[run_id] = {
        "name": f"Auto-saved: {session.target}",
        "created_at": "now",
        "target": session.target,
        "problem_type": session.problem_type,
        "models": session.models,
        "leaderboard": session.leaderboard,
        "best_model_name": session.best_model_name,
        "label_encoder": session.label_encoder,
        "feature_columns": session.feature_columns,
        "feature_importance": main._best_model_feature_importance(session.models, session.best_model_name),
    }

    summary = main._run_summary(run_id, session.saved_runs[run_id])
    assert summary["feature_importance"], "expected non-empty feature importance on auto-saved run"


def test_run_summary_falls_back_to_empty_list_for_older_runs_without_it():
    # Simulates a run saved before this field existed on disk/in memory.
    run = {
        "name": "old", "created_at": "x", "target": "t", "problem_type": "classification",
        "leaderboard": [], "best_model_name": None, "feature_columns": [],
    }
    summary = main._run_summary("rid", run)
    assert summary["feature_importance"] == []
