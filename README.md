# AutoDS — Automated Data Science Console

A working end-to-end prototype: upload a dataset, get an automated EDA profile,
run AutoML across several models, see a leaderboard with metrics and feature
importance, then make live predictions or download the trained model.

## What's in this slice

- **Backend** (`backend/`, FastAPI + pandas + scikit-learn)
  - `POST /api/upload` — upload CSV/Excel/JSON
  - `GET /api/health` — quick backend readiness check with active session count and dataset limits
  - `GET /api/eda/{session_id}` — automated profile (missingness, types, histograms, correlation)
  - `POST /api/train/{session_id}` — auto-detects classification vs. regression, trains a stronger
    model panel (linear baselines, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient
    Boosting, KNN, and optional XGBoost when installed), validates on a held-out split, returns a
    ranked leaderboard + feature importance
  - `POST /api/predict/{session_id}` — predict on new rows using the best (or chosen) model
  - `GET /api/download_model/{session_id}` — download the fitted pipeline as a `.pkl`
- **Frontend** (`frontend/index.html`) — single-page UI: drag-and-drop upload, EDA dashboard with
  charts, leaderboard table, prediction form. Talks to the backend via `fetch`, no build step needed.

Data and models live in server memory per session (a UUID) — fine for local/single-user use. For
multi-user production you'd swap that for a database + object storage (see Roadmap).

Current guardrails: uploads and remote URL datasets are capped at 200 MB, loaded datasets are capped
at 5,500,000 rows and 1,000 columns, and URL ingestion only accepts public HTTP/HTTPS addresses.

## Run it locally

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python3 -m uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser — the backend serves the frontend directly,
so there's nothing else to start.

## Try it

1. Drop in any CSV (e.g. a customer churn dataset, housing prices, fraud/risk data, or anything with a target column).
2. Review the auto-generated profile: missing values, distributions, correlations.
3. Pick a target column and click "Run AutoML" — it figures out whether it's classification or
   regression and trains/ranks models automatically.
4. Enter feature values in the Predict panel to get a live prediction, or download the best model
   as a `.pkl` to use elsewhere (`pickle.load()` gives you a dict with `pipeline`, `label_encoder`,
   `feature_columns`, `target`, `problem_type`).

### Churn example

Churn is a strong showcase use case for AutoDS. If your dataset has a target column named
`churn`, `churned`, `attrition`, `cancelled`, or similar, AutoDS will treat it as a likely
classification target. Choose that column in the Train tab to model which customers are likely
to leave, then use the Predict panel to score new customer records.

## Roadmap — turning this into the full platform

This slice covers ingestion → EDA → AutoML → predict. The fuller vision discussed includes:

- **More data sources**: database connectors (Postgres/MySQL via SQLAlchemy), API pulls, cloud
  storage (S3/GCS) — add as new `/api/connect/*` endpoints feeding the same `Session` object.
- **Deeper preprocessing controls**: let users override imputation strategy, drop columns, handle
  outliers, choose encoding — currently this is automatic; exposing it as UI options is a small
  extension of `automl.build_preprocessor`.
- **Hyperparameter tuning**: swap the fixed model panel for `GridSearchCV`/`RandomizedSearchCV` or
  a proper Bayesian search (`optuna`), per model.
- **More problem types**: time series (`statsmodels`/`prophet`), clustering (`KMeans`, silhouette
  scoring), anomaly detection.
- **Interpretability**: add SHAP value plots per prediction, partial dependence plots.
- **Persistence**: replace the in-memory `SESSIONS` dict with Postgres (metadata) + S3 (datasets/
  models), so sessions survive a restart and support multiple concurrent users.
- **Auth & multi-tenancy**: user accounts, per-user project history, sharing/collaboration.
- **Real deployment endpoint**: turn a trained model into its own hosted `/predict` API with
  versioning, rather than only a downloadable pickle.
- **Reporting**: auto-generate a PDF/notebook summary of the EDA + leaderboard + chosen model for
  sharing with non-technical stakeholders.

Each of these can be added incrementally without touching the others — the `Session` object and
the `/api/*` route structure are designed so new capabilities are new endpoints + new session
fields, not a rewrite.
