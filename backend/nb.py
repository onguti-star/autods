"""
Generate a compact Jupyter Notebook (.ipynb) for the work completed in a
session. The export intentionally avoids dumping every AutoDS helper function;
it includes only the successful user-facing results recorded on the session.
"""
import base64
import io
import json
from datetime import datetime
from typing import Any


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _json_literal(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=_json_default)


def _py_embed(value: Any) -> str:
    """
    Embed a Python value into generated notebook *source code* so it comes
    back out as the same value when the cell runs.

    _json_literal's output looks like Python (dicts/lists/strings all look
    the same in both), but it isn't: JSON's `null`/`true`/`false` are not
    valid Python — they're `None`/`True`/`False`. Any chart or result
    containing a None (e.g. a histogram with no x_min/x_max set) got
    embedded as literal `null` in the .ipynb source, which crashed with
    `NameError: name 'null' is not defined` the moment that cell ran.

    This instead serializes to a JSON *string*, double-encodes it into a
    valid Python string literal, and has the notebook parse it back with
    `json.loads(...)` at run time — so the round trip is exact regardless
    of None/True/False/nested values, and it's still just as inspectable
    since the underlying JSON is unaffected.
    """
    json_str = json.dumps(value, ensure_ascii=False, default=_json_default)
    return f"json.loads({json.dumps(json_str)})"


def _comment_lines(lines: list[str]) -> str:
    return "\n".join(f"# {line}" if line else "#" for line in lines)


def _training_runs(session) -> list[dict]:
    runs = []
    for run_id, run in (getattr(session, "saved_runs", None) or {}).items():
        runs.append({
            "run_id": run_id,
            "name": run.get("name") or f"Saved run: {run.get('target', '')}",
            "created_at": run.get("created_at"),
            "target": run.get("target"),
            "problem_type": run.get("problem_type"),
            "best_model_name": run.get("best_model_name"),
            "feature_columns": run.get("feature_columns") or [],
            "leaderboard": run.get("leaderboard") or [],
            "is_current": False,
        })

    if getattr(session, "leaderboard", None):
        runs.append({
            "run_id": "current",
            "name": f"Current training: {session.target}",
            "created_at": None,
            "target": session.target,
            "problem_type": session.problem_type,
            "best_model_name": session.best_model_name,
            "feature_columns": session.feature_columns or [],
            "leaderboard": session.leaderboard or [],
            "is_current": True,
        })
    return runs


def _has_completed_work(session, charts: list | None = None) -> bool:
    return any([
        getattr(session, "notes", None) and session.notes.strip(),
        getattr(session, "cleaning_log", None),
        getattr(session, "chat_clean_log", None),
        getattr(session, "leaderboard", None),
        getattr(session, "saved_runs", None),
        getattr(session, "saved_predictions", None),
        getattr(session, "unsupervised_results", None),
        charts,
        getattr(session, "last_visualization", None),
    ])


def _add_notes_cell(cells: list[dict], session):
    notes = (getattr(session, "notes", "") or "").strip()
    if not notes:
        return
    cells.append(_markdown_cell(f"## Notes\n\n{notes}"))


def _add_data_cell(cells: list[dict], session):
    csv_buf = io.StringIO()
    session.df.to_csv(csv_buf, index=False)
    csv_data_b64 = base64.b64encode(csv_buf.getvalue().encode("utf-8")).decode("ascii")
    cells.append(_code_cell(
        "import base64\n"
        "import io\n"
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        f"CSV_DATA_B64 = {json.dumps(csv_data_b64)}\n\n"
        "df = pd.read_csv(io.BytesIO(base64.b64decode(CSV_DATA_B64)))\n"
        "print(f'Loaded final AutoDS data: {len(df):,} rows x {len(df.columns)} columns')\n"
        "df.head()"
    ))


def _add_cleaning_cells(cells: list[dict], session):
    cleaning_log = getattr(session, "cleaning_log", None) or []
    chat_log = getattr(session, "chat_clean_log", None) or []
    if not cleaning_log and not chat_log:
        return

    lines = ["Successful cleaning/actions recorded in AutoDS:"]
    lines.extend(f"- {entry}" for entry in cleaning_log)
    lines.extend(
        f"- Chat command: {entry.get('command', '')} -> {entry.get('message', '')}"
        for entry in chat_log
    )
    cells.append(_markdown_cell("## Cleaning Done"))
    cells.append(_code_cell(
        _comment_lines(lines) + "\n\n"
        "# The exported dataframe above is already the cleaned final result.\n"
        "cleaned_df = df.copy()\n"
        "cleaned_df.to_csv('autods_cleaned_data.csv', index=False)\n"
        "print(cleaned_df.shape)\n"
    ))


def _add_visualization_cells(cells: list[dict], session, charts: list | None = None):
    # Prefer the full chart history sent from the frontend (customChartSpecs —
    # every visualization the user built for this dataset). Fall back to just
    # the most recent one if no history was passed in (e.g. old GET calls).
    all_charts = charts if charts else ([getattr(session, "last_visualization", None)] if getattr(session, "last_visualization", None) else [])
    all_charts = [c for c in all_charts if c]
    if not all_charts:
        return

    cells.append(_markdown_cell(f"## Visualizations Done ({len(all_charts)})"))
    for i, chart in enumerate(all_charts, start=1):
        cells.append(_code_cell(
            "import json\n"
            "import matplotlib.pyplot as plt\n\n"
            f"chart = {_py_embed(chart)}\n"
            "chart_type = chart.get('type')\n"
            "x = chart.get('x')\n"
            "y = chart.get('y')\n\n"
            "if chart_type in {'scatter', 'line'} and x in df.columns and y in df.columns:\n"
            "    plot_df = df[[x, y]].dropna()\n"
            "    if chart_type == 'line':\n"
            "        plot_df = plot_df.sort_values(x)\n"
            "        plt.plot(plot_df[x], plot_df[y], marker='o')\n"
            "    else:\n"
            "        plt.scatter(plot_df[x], plot_df[y], alpha=0.7)\n"
            "    plt.xlabel(x)\n"
            "    plt.ylabel(y)\n"
            "elif chart_type in {'histogram', 'boxplot'} and x in df.columns:\n"
            "    if chart_type == 'boxplot':\n"
            "        df[x].dropna().plot(kind='box')\n"
            "    else:\n"
            "        df[x].dropna().hist(bins=12)\n"
            "    plt.xlabel(x)\n"
            "elif chart_type in {'bar', 'word_frequency'} and chart.get('labels'):\n"
            "    labels = chart.get('labels', [])\n"
            "    values = chart.get('values', [])\n"
            "    plt.barh(labels[::-1], values[::-1])\n"
            "    plt.xlabel('Count')\n"
            "elif chart_type == 'wordcloud' and chart.get('words'):\n"
            "    words = chart.get('words', [])\n"
            "    labels = [w.get('word') for w in words][:25][::-1]\n"
            "    values = [w.get('count') for w in words][:25][::-1]\n"
            "    plt.barh(labels, values)\n"
            "    plt.xlabel('Count')\n"
            "elif chart_type == 'choropleth' and chart.get('rows'):\n"
            "    # Static stand-in for the interactive map: a bar chart ranking\n"
            "    # regions by the mapped value. A real choropleth would need\n"
            "    # geopandas + the boundary file, which is more than this\n"
            "    # lightweight export bundles in.\n"
            "    value_col = chart.get('value_col')\n"
            "    name_col = chart.get('name_col')\n"
            "    rows = [r for r in chart['rows'] if r.get(value_col) is not None]\n"
            "    rows.sort(key=lambda r: r[value_col], reverse=True)\n"
            "    labels = [str(r.get(name_col, '?')) for r in rows][:20][::-1]\n"
            "    values = [r[value_col] for r in rows][:20][::-1]\n"
            "    plt.barh(labels, values)\n"
            "    plt.xlabel(value_col)\n"
            "elif x in df.columns:\n"
            "    df[x].value_counts(dropna=True).head(15).sort_values().plot(kind='barh')\n"
            "    plt.xlabel('Count')\n"
            "else:\n"
            "    print('The recorded chart columns are not present in the exported data.')\n\n"
            f"plt.title(chart.get('title') or 'AutoDS chart {i}')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ))


def _add_training_cells(cells: list[dict], session):
    runs = _training_runs(session)
    if not runs:
        return

    cells.append(_markdown_cell("## Model Training Done"))
    cells.append(_code_cell(
        "import json\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n\n"
        "def train_replay(df, target, problem_type):\n"
        "    data = df.dropna(subset=[target]).copy()\n"
        "    X = data.drop(columns=[target])\n"
        "    y = data[target]\n"
        "    numeric_cols = X.select_dtypes(include='number').columns.tolist()\n"
        "    categorical_cols = [c for c in X.columns if c not in numeric_cols]\n"
        "    preprocessor = ColumnTransformer([\n"
        "        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_cols),\n"
        "        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols),\n"
        "    ])\n"
        "    if problem_type == 'classification':\n"
        "        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')\n"
        "        stratify = y if y.value_counts().min() >= 2 else None\n"
        "    else:\n"
        "        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)\n"
        "        stratify = None\n"
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)\n"
        "    pipe = Pipeline([('prep', preprocessor), ('model', model)])\n"
        "    pipe.fit(X_train, y_train)\n"
        "    preds = pipe.predict(X_test)\n"
        "    if problem_type == 'classification':\n"
        "        print('accuracy:', accuracy_score(y_test, preds))\n"
        "        print('f1_weighted:', f1_score(y_test, preds, average='weighted', zero_division=0))\n"
        "    else:\n"
        "        print('mae:', mean_absolute_error(y_test, preds))\n"
        "        print('r2:', r2_score(y_test, preds))\n"
        "    return pipe\n"
    ))

    for run in runs:
        cells.append(_code_cell(
            f"run = {_py_embed(run)}\n\n"
            "print(run['name'])\n"
            "print('target:', run['target'])\n"
            "print('problem_type:', run['problem_type'])\n"
            "print('best_model_recorded_by_autods:', run['best_model_name'])\n"
            "leaderboard = pd.DataFrame(run['leaderboard'])\n"
            "display(leaderboard)\n\n"
            "# Runnable replay using the exported final data and the same target.\n"
            "# AutoDS originally tested several models; the recorded leaderboard above is the source of truth.\n"
            "model = train_replay(df, run['target'], run['problem_type'])\n"
        ))


def _add_prediction_cells(cells: list[dict], session):
    predictions = getattr(session, "saved_predictions", None) or {}
    if not predictions:
        return

    cells.append(_markdown_cell("## Predictions Done"))
    cells.append(_code_cell(
        "import json\n\n"
        f"saved_predictions = {_py_embed(list(predictions.values()))}\n\n"
        "for prediction in saved_predictions:\n"
        "    print('target:', prediction.get('target'))\n"
        "    print('model:', prediction.get('model_name'))\n"
        "    print('inputs:', prediction.get('inputs'))\n"
        "    print('predictions:', prediction.get('predictions'))\n"
        "    print()\n"
    ))


def _add_unsupervised_cells(cells: list[dict], session):
    results = getattr(session, "unsupervised_results", None) or {}
    meaningful = {k: v for k, v in results.items() if k != "suggestions" and v}
    if not meaningful:
        return

    cells.append(_markdown_cell("## Unsupervised Analysis Done"))
    cells.append(_code_cell(
        "import json\n\n"
        f"unsupervised_results = {_py_embed(meaningful)}\n\n"
        "for name, result in unsupervised_results.items():\n"
        "    print(f'[{name}]')\n"
        "    for key, value in result.items():\n"
        "        if key in {'points', 'rules', 'rows'}:\n"
        "            print(f'{key}: {len(value) if hasattr(value, \"__len__\") else value}')\n"
        "        else:\n"
        "            print(f'{key}: {value}')\n"
        "    print()\n"
    ))


def build_notebook(session, charts: list | None = None) -> str:
    """Build a compact .ipynb JSON string for completed session work.
    `charts` is the frontend's full chart history for this dataset (same list
    used by the HTML report) so every visualization gets its own cell, not
    just the most recently created one."""
    name = session.filename.rsplit(".", 1)[0]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "AutoDS - Completed Work Notebook" if _has_completed_work(session, charts) else "AutoDS - Data Notebook"

    cells = [
        _markdown_cell(
            f"# {title}\n\n"
            f"**Dataset:** {session.filename}  \n"
            f"**Generated:** {generated_at}  \n"
            f"**Rows:** {len(session.df):,}  **Columns:** {len(session.df.columns)}\n\n"
            "This notebook keeps the export focused on the Python code and results from this session."
        )
    ]

    _add_data_cell(cells, session)
    _add_notes_cell(cells, session)
    _add_cleaning_cells(cells, session)
    _add_visualization_cells(cells, session, charts)
    _add_training_cells(cells, session)
    _add_prediction_cells(cells, session)
    _add_unsupervised_cells(cells, session)

    if len(cells) == 2:
        cells.append(_code_cell(
            "# No cleaning, training, prediction, visualization, or unsupervised actions were recorded yet.\n"
            "# The cell above contains your current dataset, ready for your own Python analysis.\n"
            "df.describe(include='all')"
        ))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False)