"""
Generate a Jupyter Notebook (.ipynb) that consolidates all AutoDS functions
for a given session — ready to upload to GitHub or run locally.
"""
import io
import json
import os
import base64
import re
from datetime import datetime

import pandas as pd

from . import clean as clean_module
from . import eda
from . import unsupervised
from . import viz
from . import pca_analysis
from . import narrate
from . import automl
from . import nlp as nlp_module
from . import assistant
from . import clean_chat


def _source_of(module) -> str:
    """Return the full source code of a module as a string."""
    import inspect
    source = inspect.getsource(module)
    # Downloaded notebooks are standalone, so package-relative imports such as
    # `from . import nlp` cannot work there. The referenced sources are embedded
    # in earlier cells instead.
    return re.sub(r"^from \.[^\n]*\n", "", source, flags=re.MULTILINE)


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


def build_notebook(session) -> str:
    """
    Build a complete .ipynb JSON string that includes all AutoDS functions
    plus the session's current data embedded as CSV so the notebook is
    self-contained and runnable on another machine (e.g. Google Colab).
    """
    cells = []

    # --- Title ---
    name = session.filename.rsplit(".", 1)[0]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    cells.append(_markdown_cell(
        f"# AutoDS — Full Analysis Notebook\n\n"
        f"**Dataset:** {session.filename}  \n"
        f"**Generated:** {generated_at}  \n"
        f"**Rows:** {len(session.df):,}  **Columns:** {len(session.df.columns)}\n\n"
        f"This notebook contains every function available in AutoDS, from data "
        f"cleaning and EDA to visualisation, PCA and AutoML training. "
        f"Each section is self-contained — run the cells in order to reproduce "
        f"your analysis or adapt them for new data.\n"
        f"---"
    ))

    # --- 1. Setup & Install ---
    cells.append(_markdown_cell(
        "## 1. Setup & Install Dependencies\n\n"
        "Run this cell once to install all required packages. "
        "If you are on Google Colab or a fresh environment, this ensures "
        "everything is available."
    ))
    cells.append(_code_cell(
        "# AutoDS — install dependencies\n"
        "# Uncomment the line below if you need to install packages:\n"
        "# !pip install pandas numpy scikit-learn openpyxl scipy joblib pyarrow\n"
        "# Optional stronger model:\n"
        "# !pip install xgboost\n\n"
        "import base64\n"
        "import io\n"
        "import json\n"
        "import re\n"
        "import types\n"
        "import warnings\n"
        "from typing import List, Tuple, Optional\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.ensemble import (\n"
        "    ExtraTreesClassifier, ExtraTreesRegressor,\n"
        "    GradientBoostingClassifier, GradientBoostingRegressor,\n"
        "    HistGradientBoostingClassifier, HistGradientBoostingRegressor,\n"
        "    RandomForestClassifier, RandomForestRegressor,\n"
        ")\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.linear_model import LogisticRegression, Ridge\n"
        "from sklearn.metrics import (\n"
        "    accuracy_score, f1_score, mean_absolute_error, precision_score,\n"
        "    r2_score, recall_score, root_mean_squared_error,\n"
        ")\n"
        "from sklearn.base import clone\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler\n"
        "from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS\n\n"
        "warnings.filterwarnings('ignore')\n"
        "print('✓ All imports ready')"
    ))

    # --- 2. Data Loading ---
    cells.append(_markdown_cell(
        "## 2. Load Data\n\n"
        "The dataset is embedded below as a CSV string so this notebook is fully "
        "self-contained — no external files needed."
    ))
    # Embed the CSV data
    csv_buf = io.StringIO()
    session.df.to_csv(csv_buf, index=False)
    csv_data_b64 = base64.b64encode(csv_buf.getvalue().encode("utf-8")).decode("ascii")
    cells.append(_code_cell(
        "# The current dataset embedded as base64-encoded CSV bytes.\n"
        "# This is safer than a raw triple-quoted CSV string for text with quotes or newlines.\n"
        f"CSV_DATA_B64 = {json.dumps(csv_data_b64)}\n\n"
        "df = pd.read_csv(io.BytesIO(base64.b64decode(CSV_DATA_B64)))\n"
        f"print(f'Loaded {{len(df):,}} rows x {{len(df.columns)}} columns: {{list(df.columns)}}')\n"
        "df.head()"
    ))

    # Shared text helpers are embedded before EDA because several later modules
    # reference them through a lightweight `nlp.*` namespace.
    cells.append(_markdown_cell(
        "## 3. NLP Utilities\n\n"
        "Functions for analysing text columns — word frequency, vocabulary size, etc."
    ))
    cells.append(_code_cell(
        _source_of(nlp_module)
    ))
    cells.append(_code_cell(
        "# Compatibility namespace for embedded modules that refer to `nlp.*`.\n"
        "nlp = types.SimpleNamespace(\n"
        "    SKLEARN_STOPWORDS=SKLEARN_STOPWORDS,\n"
        "    is_text_column=is_text_column,\n"
        "    text_column_stats=text_column_stats,\n"
        "    word_frequency=word_frequency,\n"
        ")\n"
    ))

    # --- 3. EDA Module (all functions) ---
    cells.append(_markdown_cell(
        "## 4. Exploratory Data Analysis (EDA)\n\n"
        "All functions from the EDA module — profile your data, check correlations, "
        "and get a JSON-serialisable summary without writing pandas code."
    ))
    cells.append(_code_cell(
        _source_of(eda)
    ))
    cells.append(_code_cell(
        "# Use EDA functions\n"
        "profile = profile_dataframe(df)\n"
        "print(f'Shape: {profile[\"shape\"]}')\n"
        "print(f'Duplicate rows: {profile[\"duplicate_rows\"]:,}')\n"
        "print(f'Missing cells: {profile[\"total_missing_cells\"]:,}')\n"
        "print()\n"
        "for col in profile['columns']:\n"
        "    print(f'  {col[\"name\"]:25s} | {col[\"type\"]:12s} | missing {col[\"missing_pct\"]:>6.2f}% | unique {col[\"unique\"]:,}')\n\n"
        "# Preview first 10 rows\n"
        "preview = safe_preview(df, 10)\n"
        "print(f'\\nFirst {len(preview)} rows:')\n"
        "display(pd.DataFrame(preview))"
    ))

    # --- 4. NLP Module ---
    cells.append(_markdown_cell(
        "## 5. Text Column Analysis\n\n"
        "Use the NLP utilities to inspect text columns in your dataset."
    ))
    cells.append(_code_cell(
        "# Find text columns and run NLP analysis\n"
        "text_cols = [c for c in df.columns if is_text_column(df[c])]\n"
        "if text_cols:\n"
        "    for c in text_cols:\n"
        "        stats = text_column_stats(df[c])\n"
        "        print(f\"'{c}': {stats['documents']:,} docs, {stats['vocab_size']:,} unique words, \"\n"
        "              f\"avg {stats['avg_words']:.1f} words/doc\")\n"
        "        top = word_frequency(df[c], top_n=10)\n"
        "        print(f\"  Top words: {', '.join(w['word'] for w in top)}\")\n"
        "else:\n"
        "    print('No text columns detected.')"
    ))

    # --- 5. Narrate Module ---
    cells.append(_markdown_cell(
        "## 6. Plain-English Narration\n\n"
        "Turn the numeric EDA profile into human-readable explanations."
    ))
    cells.append(_code_cell(
        _source_of(narrate)
    ))
    cells.append(_code_cell(
        "# Generate narrative explanations\n"
        "profile = profile_dataframe(df)\n"
        "narrative = narrate_eda(profile)\n"
        "print('### EDA Summary ###')\n"
        "for note in narrative:\n"
        "    print(f'• {note}')"
    ))

    # --- 6. Data Cleaning Module ---
    cells.append(_markdown_cell(
        "## 7. Data Cleaning\n\n"
        "Comprehensive cleaning pipeline — duplicate removal, missing value imputation, "
        "outlier capping, type coercion, and more. All functions from `clean.py`."
    ))
    cells.append(_code_cell(
        _source_of(clean_module)
    ))
    cells.append(_code_cell(
        "# Run the full cleaning pipeline\n"
        "options = {\n"
        "    'drop_duplicates': True,\n"
        "    'drop_constant_cols': True,\n"
        "    'drop_high_missing_cols': True,\n"
        "    'strip_whitespace': True,\n"
        "    'fix_mixed_types': True,\n"
        "    'fix_column_names': True,\n"
        "    'remove_outliers': True,\n"
        "    'fill_missing_numeric': 'median',\n"
        "    'fill_missing_categorical': 'mode',\n"
        "}\n\n"
        "cleaned_df, clean_log = clean_dataframe(df.copy(), options)\n"
        "for entry in clean_log:\n"
        "    print(f'✓ {entry}')\n\n"
        "print(f'\\nBefore: {len(df):,} rows x {len(df.columns)} cols')\n"
        "print(f'After:  {len(cleaned_df):,} rows x {len(cleaned_df.columns)} cols')"
    ))

    # --- 7. Visualisation Module ---
    cells.append(_markdown_cell(
        "## 8. Visualisation\n\n"
        "Generate Chart.js-ready data for all chart types — histograms, bar charts, "
        "scatter plots, box plots, pie charts, line charts, bubble charts, word clouds, and more."
    ))
    cells.append(_code_cell(
        _source_of(viz)
    ))
    cells.append(_code_cell(
        "# Get auto-suggested charts\n"
        "suggestions = suggest_visuals(df)\n"
        "print(f'Auto-suggested {len(suggestions)} chart(s):')\n"
        "for s in suggestions:\n"
        "    print(f'  • {s[\"title\"]}')\n"
        "    print(f'    Reason: {s[\"reason\"]}')\n\n"
        "# You can also build custom charts:\n"
        "# chart_data(df, 'column_name', 'histogram')\n"
        "# chart_data(df, 'column_name', 'bar')\n"
        "# chart_data(df, 'column_x', 'scatter', y='column_y')\n"
        "# chart_data(df, 'column_x', 'line', y='column_y')"
    ))

    # --- 8. PCA Module ---
    cells.append(_markdown_cell(
        "## 9. Principal Component Analysis (PCA)\n\n"
        "Check whether numeric columns can be compressed into fewer dimensions."
    ))
    cells.append(_code_cell(
        _source_of(pca_analysis)
    ))
    cells.append(_code_cell(
        "# Run PCA analysis\n"
        "result = analyse(df)\n"
        "if result.get('feasible'):\n"
        "    print(f'PCA Verdict: {result[\"verdict\"]}')\n"
        "    print(f'Numeric columns: {result[\"n_numeric\"]}')\n"
        "    print(f'Components for 80% variance: {result[\"components_for_80\"]}')\n"
        "    print(f'Components for 90% variance: {result[\"components_for_90\"]}')\n"
        "    print(f'Top 2 PCs capture: {result[\"top2_variance\"]}% variance')\n"
        "    print(f'\\nRecommendation: {result[\"recommendation\"]}')\n"
        "else:\n"
        "    print(f'PCA not feasible: {result.get(\"reason\", \"\")}')"
    ))

    # --- 9. AutoML Module ---
    cells.append(_markdown_cell(
        "## 10. AutoML Training\n\n"
        "Automatically detect classification vs regression and train a stronger panel of model types. "
        "Includes cross-validation, leaderboard ranking, and feature importance extraction."
    ))
    cells.append(_code_cell(
        _source_of(automl)
    ))

    # Build session-specific training cells — one per trained target
    all_nb_runs = []
    for run_id, run in (session.saved_runs or {}).items():
        all_nb_runs.append({
            "target": run.get("target", ""),
            "problem_type": run.get("problem_type", ""),
            "best_model_name": run.get("best_model_name", ""),
            "leaderboard": run.get("leaderboard", []),
            "feature_columns": run.get("feature_columns", []),
            "label": run.get("name", f"Auto-saved: {run.get('target', '?')}"),
            "is_current": False,
        })
    if session.leaderboard:
        all_nb_runs.append({
            "target": session.target or "",
            "problem_type": session.problem_type or "",
            "best_model_name": session.best_model_name or "",
            "leaderboard": session.leaderboard,
            "feature_columns": session.feature_columns or [],
            "label": f"Current model — {session.target}",
            "is_current": True,
        })

    if all_nb_runs:
        # Intro cell describing what was trained in this session
        targets_summary = ", ".join(f"`{r['target']}`" for r in all_nb_runs)
        cells.append(_markdown_cell(
            f"### Session Training Results\n\n"
            f"In this AutoDS session, **{len(all_nb_runs)} model(s)** were trained: {targets_summary}.\n\n"
            f"The cells below reproduce each training run. Run them to retrain the same models "
            f"on the embedded dataset, or change `TARGET` to experiment with a different column."
        ))

        for run in all_nb_runs:
            target = run["target"]
            feat_cols = run["feature_columns"]
            problem_type = run["problem_type"]
            best_model = run["best_model_name"]
            leaderboard = run["leaderboard"]
            label = run["label"]

            # Build a summary comment showing what AutoDS found
            summary_lines = [f"# {label}"]
            summary_lines.append(f"# Problem type: {problem_type}")
            summary_lines.append(f"# Best model:   {best_model}")
            if leaderboard:
                summary_lines.append("# Leaderboard:")
                for i, row in enumerate(leaderboard):
                    marker = " ⭐" if row.get("model") == best_model else ""
                    if row.get("error"):
                        summary_lines.append(f"#   {i+1}. {row.get('model', '?')}{marker} — failed: {row['error']}")
                    else:
                        metrics_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                for k, v in row.get("metrics", {}).items())
                        summary_lines.append(f"#   {i+1}. {row.get('model', '?')}{marker}: {metrics_str}")

            # Feature-column line
            if feat_cols:
                feat_repr = repr(feat_cols)
                drop_cols_code = f"X = df[{feat_repr}]"
            else:
                drop_cols_code = f"X = df.drop(columns=[TARGET])"

            cells.append(_markdown_cell(
                f"#### Train on `{target}` ({label})\n\n"
                f"Problem type: **{problem_type}** · Best model: **{best_model}**"
            ))
            cells.append(_code_cell(
                "\n".join(summary_lines) + "\n\n"
                f"TARGET = {repr(target)}\n"
                f"print(f'Training models to predict: {{TARGET}}')\n\n"
                f"try:\n"
                f"    problem_type, leaderboard, fitted, best_name, label_encoder = train_all(df, TARGET)\n"
                f"    print(f'\\nProblem type: {{problem_type}}')\n"
                f"    print(f'Best model:   {{best_name}}')\n"
                f"    print()\n"
                f"    for row in leaderboard:\n"
                f"        if 'metrics' in row:\n"
                f"            print(f'  {{row[\"model\"]:30s}} → {{row[\"metrics\"]}}')\n"
                f"        else:\n"
                f"            print(f'  {{row[\"model\"]:30s}} ✗ failed: {{row[\"error\"]}}')\n\n"
                f"    # Feature importance\n"
                f"    {drop_cols_code}\n"
                f"    importance = feature_importance(fitted[best_name], X) if best_name and best_name in fitted else []\n"
                f"    if importance:\n"
                f"        print(f'\\nTop features ({{best_name}}):')\n"
                f"        for item in importance[:10]:\n"
                f"            print(f'  {{item[\"feature\"]:30s}} importance = {{item[\"importance\"]:.4f}}')\n\n"
                f"    # Predict — fill in values for each feature\n"
                f"    sample_row = {{col: None for col in {repr(feat_cols)}}}\n"
                f"    # example: sample_row = {{{', '.join(repr(c)+': 0' for c in (feat_cols[:3] if feat_cols else []))}}}\n"
                f"    import pandas as _pd\n"
                f"    _X_sample = _pd.DataFrame([sample_row])\n"
                f"    _pred = fitted[best_name].predict(_X_sample)\n"
                f"    print(f'\\nSample prediction for {{TARGET}}: {{_pred[0]}}')\n\n"
                f"except ValueError as e:\n"
                f"    print(f'Training error: {{e}}')"
            ))
    else:
        # No training done — keep the generic placeholder
        cells.append(_code_cell(
            "# Run AutoML — pick your target column\n"
            "TARGET = df.columns[-1]  # Change this to your desired target column\n"
            "print(f'Training models to predict: {TARGET}')\n\n"
            "try:\n"
            "    problem_type, leaderboard, fitted, best_name, label_encoder = train_all(df, TARGET)\n"
            "    print(f'\\nProblem type: {problem_type}')\n"
            "    print(f'Best model: {best_name}')\n"
            "    print()\n"
            "    for row in leaderboard:\n"
            "        if 'metrics' in row:\n"
            "            print(f'  {row[\"model\"]:30s} → {row[\"metrics\"]}')\n"
            "        else:\n"
            "            print(f'  {row[\"model\"]:30s} ✗ failed: {row[\"error\"]}')\n"
            "except ValueError as e:\n"
            "    print(f'Training error: {e}')"
        ))

    # --- 10. Training Narrative ---
    cells.append(_markdown_cell(
        "## 11. Training Narrative\n\n"
        "Get plain-English explanations of your model training results."
    ))
    if all_nb_runs:
        for run in all_nb_runs:
            target = run["target"]
            feat_cols = run["feature_columns"]
            drop_code = f"X = df[{repr(feat_cols)}]" if feat_cols else f"X = df.drop(columns=[TARGET])"
            cells.append(_code_cell(
                f"# Training narrative for '{target}'\n"
                f"TARGET = {repr(target)}\n"
                f"try:\n"
                f"    problem_type, leaderboard, fitted, best_name, label_encoder = train_all(df, TARGET)\n"
                f"    {drop_code}\n"
                f"    importance = feature_importance(fitted[best_name], X) if (best_name and best_name in fitted) else []\n"
                f"    training_note = narrate_training(problem_type, TARGET, leaderboard, best_name, importance)\n"
                f"    print(f'### Training Summary for {{TARGET}} ###')\n"
                f"    for note in training_note:\n"
                f"        print(f'• {{note}}')\n"
                f"except ValueError as e:\n"
                f"    print(f'Could not train: {{e}}')"
            ))
    else:
        cells.append(_code_cell(
            "# Generate training narrative\n"
            "TARGET = df.columns[-1]\n"
            "try:\n"
            "    problem_type, leaderboard, fitted, best_name, label_encoder = train_all(df, TARGET)\n"
            "    X = df.drop(columns=[TARGET])\n"
            "    importance = feature_importance(fitted[best_name], X) if (best_name and best_name in fitted) else []\n"
            "    training_note = narrate_training(problem_type, TARGET, leaderboard, best_name, importance)\n"
            "    print('### Training Summary ###')\n"
            "    for note in training_note:\n"
            "        print(f'• {note}')\n"
            "except ValueError as e:\n"
            "    print(f'Could not train: {e}')"
        ))

    # --- 11. Assistant / Question Answering ---
    cells.append(_markdown_cell(
        "## 12. Ask Questions About Your Data\n\n"
        "The assistant module can answer questions about your dataset using "
        "programmatic analysis — no external API calls needed."
    ))
    cells.append(_code_cell(
        _source_of(assistant)
    ))
    cells.append(_code_cell(
        "# Ask questions about your data\n"
        "questions = [\n"
        "    'describe this dataset',\n"
        "    'strongest correlations',\n"
        "    'outliers in numeric columns',\n"
        "    'summary statistics',\n"
        "]\n\n"
        "for q in questions:\n"
        "    print(f'Q: {q}')\n"
        "    answer = answer_question(df, q)\n"
        "    print(f'A: {answer}')\n"
        "    print()"
    ))

    # --- 12. Clean Chat Module ---
    cells.append(_markdown_cell(
        "## 13. Interactive Cleaning Commands\n\n"
        "Use natural language commands to clean your data — like 'remove duplicates', "
        "'drop column X', 'trim whitespace', etc."
    ))
    cells.append(_code_cell(
        _source_of(clean_chat)
    ))
    cells.append(_code_cell(
        "# Interactive cleaning via natural language commands\n"
        "# Try commands like:\n"
        "#   - 'remove duplicates'\n"
        "#   - 'drop column column_name'\n"
        "#   - 'trim whitespace'\n"
        "#   - 'remove outliers'\n"
        "#   - 'standardize column names'\n"
        "#   - 'convert column_name to numeric'\n"
        "#   - 'help' (see all available commands)\n\n"
        "example_cmd = 'remove duplicates'\n"
        "modified_df, message = run_command(df.copy(), example_cmd)\n"
        "print(f'Command: \"{example_cmd}\"')\n"
        "print(f'Result: {message}')\n\n"
        "print(f'\\nFull help text:\\n{HELP_TEXT}')"
    ))

    # --- 13. Clean Dataframe (the main cleaning entry point) ---
    cells.append(_markdown_cell(
        "## 14. Cleaning Recommendations\n\n"
        "Get a smart cleaning plan tailored to your dataset before applying transformations."
    ))
    cells.append(_code_cell(
        "# Get cleaning recommendations\n"
        "recs = build_cleaning_recommendations(df)\n"
        "print(f'Summary: {recs[\"summary\"]}')\n"
        "print(f'\\nIssues found ({len(recs[\"issues\"])}):')\n"
        "for issue in recs['issues']:\n"
        "    print(f'  [{issue[\"priority\"]}] {issue[\"title\"]}')\n"
        "    print(f'       {issue[\"detail\"]}')\n"
        "print(f'\\nRecommended options: {json.dumps(recs[\"recommended_options\"], indent=2)}')"
    ))

    # --- 14. Correlation Matrix ---
    cells.append(_markdown_cell(
        "## 15. Correlation Analysis\n\n"
        "Detailed correlation matrix between all numeric columns."
    ))
    cells.append(_code_cell(
        "# Compute and display the correlation matrix\n"
        "corr = correlation_matrix(df)\n"
        "if corr['matrix'] and len(corr['columns']) >= 2:\n"
        "    import numpy as np\n"
        "    corr_df = pd.DataFrame(corr['matrix'],\n"
        "                             index=corr['columns'],\n"
        "                             columns=corr['columns'])\n"
        "    display(corr_df.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))\n\n"
        "    # Find strongest correlations\n"
        "    pairs = []\n"
        "    for i, a in enumerate(corr['columns']):\n"
        "        for j in range(i + 1, len(corr['columns'])):\n"
        "            pairs.append((a, corr['columns'][j], corr['matrix'][i][j]))\n"
        "    pairs.sort(key=lambda x: abs(x[2]), reverse=True)\n"
        "    print('\\nTop 5 strongest correlations:')\n"
        "    for a, b, v in pairs[:5]:\n"
        "        direction = 'positive' if v > 0 else 'negative'\n"
        "        print(f'  {a:20s} ↔ {b:20s} : {v:+.3f} ({direction})')\n"
        "else:\n"
        "    print('Not enough numeric columns for correlation analysis.')"
    ))

    # --- 15. Unsupervised Learning Module ---
    cells.append(_markdown_cell(
        "## 16. Unsupervised Learning\n\n"
        "Discover patterns without a target column — clustering, anomaly detection, "
        "dimensionality reduction, and association rules."
    ))
    cells.append(_code_cell(
        _source_of(unsupervised)
    ))
    cells.append(_code_cell(
        "# Get suggestions for which unsupervised methods to use\n"
        "suggestions = suggest_unsupervised(df)\n"
        "print(f'Dataset has {suggestions[\"n_numeric\"]} numeric and {suggestions[\"n_categorical\"]} categorical columns')\n"
        "print('\\nSuggested unsupervised tasks:')\n"
        "for s in suggestions['suggestions']:\n"
        "    print(f'  • {s[\"task\"]}: {s[\"reason\"]}')\n"
        "    print(f'    Methods: {', '.join(s['methods'])}')\n\n"
        "# Example: K-Means clustering\n"
        "# result = cluster_kmeans(df, n_clusters=3)\n"
        "# print(f'Clusters: {result[\"cluster_sizes\"]}')\n"
        "# print(f'Metrics: {result[\"metrics\"]}')\n\n"
        "# Example: Anomaly detection\n"
        "# result = detect_anomalies_isolation_forest(df, contamination=0.1)\n"
        "# print(f'Anomalies found: {result[\"n_outliers\"]} ({result[\"outlier_percentage\"]}%)')\n\n"
        "# Example: Dimensionality reduction\n"
        "# result = reduce_tsne(df, n_components=2, perplexity=30.0)\n"
        "# print(f't-SNE generated {len(result[\"points\"])} 2D points')\n\n"
        "# Example: Association rules (requires mlxtend)\n"
        "# result = association_rules(df, min_support=0.1, min_confidence=0.5)\n"
        "# print(f'Found {result[\"n_rules\"]} rules')\n"
        "# for rule in result['rules'][:5]:\n"
        "#     print(f'  {rule[\"antecedents\"]} -> {rule[\"consequents\"]} (lift: {rule[\"lift\"]})')\n"
    ))

    # --- Footer ---
    cells.append(_markdown_cell(
        "---\n\n"
        "*This notebook was automatically generated by [AutoDS](https://github.com/your-org/autods). "
        "All functions are self-contained — you can modify, extend, and re-use them freely.*"
    ))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells,
    }

    return json.dumps(notebook, indent=1, ensure_ascii=False)