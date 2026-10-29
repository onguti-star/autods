import pandas as pd

from backend.store import Session


def test_clear_current_training_auto_preserves_model():
    """Clearing current training should auto-snapshot the model into saved_runs
    so it is never lost when the data changes or the user switches targets."""
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

    # The previously trained model should now be preserved as a saved run
    assert "run_1" in session.saved_runs  # existing saved run untouched
    auto_saved = [r for r in session.saved_runs.values() if r.get("auto_saved")]
    assert len(auto_saved) == 1
    assert auto_saved[0]["target"] == "y"
    assert auto_saved[0]["best_model_name"] == "model"
    # The snapshot must keep its own reference to the models dict
    assert "model" in auto_saved[0]["models"]
    assert auto_saved[0]["feature_columns"] == ["x"]


def test_clear_current_training_no_models():
    """Clearing with no trained models should not create a saved run."""
    session = Session(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "data.csv")
    session.clear_current_training()
    assert session.saved_runs == {}


def test_clear_current_training_no_target():
    """Clearing with models but no target set should not create a saved run."""
    session = Session(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "data.csv")
    session.models = {"model": object()}
    session.target = None
    session.clear_current_training()
    assert session.saved_runs == {}


def test_auto_preserve_can_be_disabled():
    """Passing auto_preserve=False should skip the snapshot (teardown case)."""
    session = Session(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "data.csv")
    session.target = "y"
    session.models = {"model": object()}
    session.clear_current_training(auto_preserve=False)
    assert session.saved_runs == {}


def test_repeated_clear_does_not_duplicate_runs():
    """Clearing repeatedly (e.g. consecutive data changes) should only create
    one auto-saved run for a given (target, best_model) pair."""
    session = Session(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "data.csv")
    session.target = "y"
    session.problem_type = "regression"
    session.models = {"model": object()}
    session.best_model_name = "model"

    session.clear_current_training()
    # Re-train the same target + best model, then clear again
    session.models = {"model": object()}
    session.best_model_name = "model"
    session.clear_current_training()

    auto_saved = [r for r in session.saved_runs.values() if r.get("auto_saved")]
    assert len(auto_saved) == 1


def test_automatic_saved_runs_are_bounded():
    session = Session(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "data.csv")

    for index in range(Session.MAX_AUTO_SAVED_RUNS + 2):
        session.add_saved_run(
            f"auto_{index}",
            {"auto_saved": True, "name": f"Auto {index}"},
        )

    automatic = [run for run in session.saved_runs.values() if run.get("auto_saved")]
    assert len(automatic) == Session.MAX_AUTO_SAVED_RUNS
    assert "auto_0" not in session.saved_runs


def test_undo_snapshots_respect_memory_budget(monkeypatch):
    session = Session(pd.DataFrame({"value": list(range(100))}), "data.csv")
    monkeypatch.setattr(Session, "UNDO_LIMIT", 20)
    monkeypatch.setattr(Session, "UNDO_MAX_BYTES", 1)

    session.snapshot_before_change()

    assert not session.can_undo()


def test_invalidate_training_on_schema_change_clears_stale_columns():
    """If a cleaned dataset renames or drops columns, any stale training state
    for the previous schema must be discarded immediately so re-training uses
    the current dataframe instead of ghost columns from the earlier dataset."""
    session = Session(pd.DataFrame({"age": [20, 25], "income": [50_000, 60_000], "target": [1, 0]}), "data.csv")
    session.target = "target"
    session.problem_type = "classification"
    session.models = {"model": object()}
    session.best_model_name = "model"
    session.feature_columns = ["age", "income", "old_feature"]
    session.label_encoder = object()

    session.df = session.df.rename(columns={"income": "salary"}).drop(columns=["age"])
    session.invalidate_training_if_schema_changed()

    assert session.target is None
    assert session.models == {}
    assert session.best_model_name is None
    assert session.feature_columns == []
    assert session.label_encoder is None
    assert session.progress_messages == []
