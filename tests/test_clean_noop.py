import pandas as pd

from backend import main, store


def test_clean_noop_does_not_create_undo_or_clear_training():
    session = store.create_session(
        pd.DataFrame({"age": [20, 30, 40], "city": ["Nairobi", "Mombasa", "Kisumu"]}),
        "clean.csv",
    )
    existing_model = object()
    session.models = {"existing": existing_model}
    session.target = "age"
    session.problem_type = "regression"
    session.best_model_name = "existing"

    result = main.clean_data(
        session.id,
        main.CleanRequest(
            drop_duplicates=False,
            fill_missing_numeric="none",
            fill_missing_categorical="none",
            drop_high_missing_cols=False,
            strip_whitespace=False,
            drop_constant_cols=False,
            fix_mixed_types=False,
            fix_column_names=False,
            remove_outliers=False,
        ),
    )

    assert result["data_changed"] is False
    assert result["can_undo"] is False
    assert session.models == {"existing": existing_model}
    assert session.target == "age"


def test_json_safe_records_accepts_list_like_cells():
    df = pd.DataFrame({"tags": [["a", "b"], None], "value": [1, float("nan")]})

    rows = main._json_safe_records(df)

    assert rows[0]["tags"] == ["a", "b"]
    assert rows[0]["value"] == 1
    assert rows[1]["tags"] is None
    assert rows[1]["value"] is None
