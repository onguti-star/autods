"""
AutoML core: given a dataframe + target column, this:
  1. Infers the problem type (classification vs regression)
  2. Builds a preprocessing pipeline (impute, scale, one-hot encode)
  3. Trains a panel of candidate models
  4. Cross-validates and ranks them on a held-out test split
  5. Returns a leaderboard + keeps fitted pipelines for prediction/export
"""
import math
import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

from . import nlp

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # XGBoost is optional; scikit-learn models remain the default fallback.
    XGBClassifier = None
    XGBRegressor = None


LARGE_DATASET_THRESHOLD = 100_000
LARGE_DATASET_MIN_SAMPLE = 20_000
LARGE_DATASET_MAX_SAMPLE = 75_000
LARGE_DATASET_SAMPLE_FRACTION = 0.05


def _large_dataset_sample_size(n_rows: int) -> int:
    if n_rows <= LARGE_DATASET_THRESHOLD:
        return n_rows
    adaptive_size = math.ceil(n_rows * LARGE_DATASET_SAMPLE_FRACTION)
    return min(n_rows, max(LARGE_DATASET_MIN_SAMPLE, min(LARGE_DATASET_MAX_SAMPLE, adaptive_size)))


def infer_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        # numeric but few unique values -> likely classification (e.g. 0/1, star ratings)
        n_unique = y.nunique(dropna=True)
        if n_unique <= max(10, int(0.05 * len(y))) and n_unique < 20:
            return "classification"
        return "regression"
    return "classification"


def _safe_transformer_name(prefix: str, index: int, col: str) -> str:
    """sklearn forbids '__' in transformer names (reserved for nested param access).
    Keeps the column name for readability in feature importance output, sanitized
    and de-duplicated of underscores, with an index prefix to guarantee uniqueness."""
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", str(col)).strip("_")
    name = f"{prefix}{index}_{safe}" if safe else f"{prefix}{index}"
    return re.sub(r"_+", "_", name)


def _flatten_text_column(X):
    """ColumnTransformer passes a single text column as a 2D (n,1) slice;
    TfidfVectorizer needs a 1D iterable of strings. Also fills missing text
    with an empty string rather than dropping/erroring on it.
    Must be a module-level function (not a lambda) so the fitted pipeline —
    and therefore the whole model — can still be pickled for download."""
    return X.iloc[:, 0].fillna("").astype(str)


def _to_dense(X):
    """Some high-performing sklearn estimators do not accept sparse matrices."""
    if hasattr(X, "toarray"):
        arr = X.toarray()
        return arr.astype(np.float32, copy=False)
    return X


def build_preprocessor(
    X: pd.DataFrame,
    use_pca: bool = False,
    *,
    max_text_features: int = 300,
    one_hot_min_frequency: int | None = None,
) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    remaining_cols = [c for c in X.columns if c not in numeric_cols]
    text_cols = [c for c in remaining_cols if nlp.is_text_column(X[c])]
    categorical_cols = [c for c in remaining_cols if c not in text_cols]

    numeric_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if use_pca and len(numeric_cols) >= 2:
        numeric_steps.append(("pca", PCA(n_components=0.95, random_state=42)))
    
    numeric_pipe = Pipeline(steps=numeric_steps)
    categorical_kwargs = {
        "handle_unknown": "ignore",
        "sparse_output": True,
        "dtype": np.float32,
    }
    if one_hot_min_frequency is not None:
        categorical_kwargs["min_frequency"] = one_hot_min_frequency
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**categorical_kwargs)),
    ])

    transformers = [
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ]
    for i, col in enumerate(text_cols):
        text_pipe = Pipeline(steps=[
            ("flatten", FunctionTransformer(_flatten_text_column, feature_names_out="one-to-one")),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_text_features,
                    stop_words=nlp.SKLEARN_STOPWORDS,
                    ngram_range=(1, 1),
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ])
        # sklearn forbids "__" in transformer names (reserved for nested param access),
        # so an index-based name is used rather than embedding the raw column name.
        transformers.append((_safe_transformer_name("text", i, col), text_pipe, [col]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def _dense_model(model):
    return Pipeline(steps=[
        ("dense", FunctionTransformer(_to_dense, accept_sparse=True)),
        ("estimator", model),
    ])


def _train_single_model(name, model, X_train, X_test, y_train, y_test, problem_type, preprocessor, n_neighbors):
    """Train a single model and return results."""
    try:
        candidate_model = clone(model)
        if name == "K-Nearest Neighbors":
            candidate_model.set_params(n_neighbors=n_neighbors)
        pipe = Pipeline(steps=[("prep", clone(preprocessor)), ("model", candidate_model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        if problem_type == "classification":
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "f1_weighted": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
                "precision_weighted": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
                "recall_weighted": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4),
            }
            primary = metrics["f1_weighted"]
        else:
            rmse = float(root_mean_squared_error(y_test, preds))
            metrics = {
                "rmse": round(rmse, 4),
                "mae": round(float(mean_absolute_error(y_test, preds)), 4),
                "r2": round(float(r2_score(y_test, preds)), 4),
            }
            primary = metrics["r2"]

        return {"model": name, "metrics": metrics, "primary_score": primary, "fitted": pipe}
    except Exception as e:
        return {"model": name, "error": str(e)}


def _classification_models(n_classes: int) -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Histogram Gradient Boosting": _dense_model(HistGradientBoostingClassifier(random_state=42)),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }
    if XGBClassifier is not None:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective=objective,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    return models


def _regression_models() -> dict:
    models = {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Histogram Gradient Boosting": _dense_model(HistGradientBoostingRegressor(random_state=42)),
        "K-Nearest Neighbors": KNeighborsRegressor(),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    return models


def _fast_classification_models(n_classes: int) -> dict:
    """Fast models optimized for large datasets (>100k rows)."""
    models = {
        "Linear Classifier (Fast)": SGDClassifier(
            loss="log_loss",
            max_iter=300,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_jobs=-1,
            random_state=42,
        ),
    }
    if XGBClassifier is not None:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        models["XGBoost (Fast)"] = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective=objective,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return models


def _fast_regression_models() -> dict:
    """Fast models optimized for large datasets (>100k rows)."""
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
    }
    if XGBRegressor is not None:
        models["XGBoost (Fast)"] = XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return models


def train_all(
    df: pd.DataFrame,
    target: str,
    use_pca: bool = False,
    progress_callback=None,
) -> Tuple[str, list, dict, str, object]:
    """
    Returns: (problem_type, leaderboard, fitted_pipelines_by_name, best_name, label_encoder_or_None)
    
    If use_pca=True, applies PCA to numeric features before training to reduce dimensionality.
    Automatically optimizes for large datasets (>100k rows) by sampling and using faster models.

    progress_callback: optional callable(str) -> None, invoked with a short status
    message before each candidate model starts training. Lets the caller (e.g. a
    background training job) surface live progress to the UI.
    """
    valid_target = df[target].notna()
    if not valid_target.any():
        raise ValueError(f"Target column '{target}' has no non-missing values.")

    y_raw = df.loc[valid_target, target]
    if len(df.columns) <= 1:
        raise ValueError("Training needs at least one feature column besides the target.")
    if len(y_raw) < 5:
        raise ValueError("Training needs at least 5 rows with a non-missing target.")

    problem_type = infer_problem_type(y_raw)

    label_encoder = None
    if problem_type == "classification":
        class_counts = y_raw.astype(str).value_counts()
        if len(class_counts) < 2:
            raise ValueError(f"Target column '{target}' has only one class. Choose a target with at least two values.")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
    else:
        if pd.to_numeric(y_raw, errors="coerce").isna().any():
            raise ValueError(f"Target column '{target}' contains non-numeric values, so it cannot be used for regression.")
        if y_raw.nunique(dropna=True) < 2:
            raise ValueError(f"Target column '{target}' has only one unique value. Choose a target with variation.")
        y = pd.to_numeric(y_raw, errors="raise").to_numpy()

    # OPTIMIZATION: Sample data for large datasets.
    # Pick row positions before building X so 1M+ row jobs do not copy and
    # preprocess the entire dataframe just to discard most rows afterward.
    original_n_rows = len(y_raw)
    n_rows = original_n_rows
    is_large_dataset = original_n_rows > LARGE_DATASET_THRESHOLD
    
    if is_large_dataset:
        sample_size = _large_dataset_sample_size(n_rows)
        if progress_callback:
            try:
                progress_callback(
                    f"Large dataset detected ({original_n_rows:,} rows). "
                    f"Training on a representative sample of {sample_size:,} rows..."
                )
            except Exception:
                pass
        
        if problem_type == "classification":
            # Stratified sampling to preserve class distribution
            stratify_sample = y if pd.Series(y).value_counts().min() >= 2 else None
            _, sample_idx = train_test_split(
                np.arange(n_rows),
                test_size=sample_size,
                random_state=42,
                stratify=stratify_sample,
            )
        else:
            # Random sampling for regression
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(n_rows, size=sample_size, replace=False)
        
        X = df.loc[y_raw.index[sample_idx]].drop(columns=[target]).reset_index(drop=True)
        y = y[sample_idx]
        n_rows = sample_size
    else:
        X = df.loc[y_raw.index].drop(columns=[target]).reset_index(drop=True)

    stratify = y if problem_type == "classification" and pd.Series(y).value_counts().min() >= 2 else None
    test_size = max(1, int(round(n_rows * (0.1 if is_large_dataset else 0.2))))
    if stratify is not None:
        n_classes = len(np.unique(y))
        test_size = max(test_size, n_classes)
        if n_rows - test_size < n_classes:
            stratify = None
            test_size = max(1, int(round(n_rows * 0.2)))
    if n_rows - test_size < 1:
        raise ValueError("Training needs enough rows to create both train and test samples.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
    except ValueError as e:
        raise ValueError(f"Could not split the dataset for training: {e}") from e

    preprocessor = build_preprocessor(
        X,
        use_pca=use_pca,
        max_text_features=120 if is_large_dataset else 300,
        one_hot_min_frequency=10 if is_large_dataset else None,
    )
    
    # OPTIMIZATION: Use faster models for large datasets
    if is_large_dataset:
        candidates = (
            _fast_classification_models(len(np.unique(y)))
            if problem_type == "classification"
            else _fast_regression_models()
        )
    else:
        candidates = (
            _classification_models(len(np.unique(y)))
            if problem_type == "classification"
            else _regression_models()
        )

    n_neighbors = min(5, len(X_train))
    
    # OPTIMIZATION: Train models sequentially for large datasets to avoid memory issues
    # Parallel processing can cause memory overflow with large datasets
    results = []
    for name, model in candidates.items():
        if progress_callback:
            try:
                progress_callback(f"Training {name}...")
            except Exception:
                pass  # never let a progress-reporting hiccup break training
        result = _train_single_model(
            name, model, X_train, X_test, y_train, y_test, problem_type, preprocessor, n_neighbors
        )
        results.append(result)

    fitted = {}
    leaderboard = []
    
    for result in results:
        if "error" in result:
            leaderboard.append({"model": result["model"], "error": result["error"]})
        else:
            fitted[result["model"]] = result["fitted"]
            leaderboard.append({
                "model": result["model"],
                "metrics": result["metrics"],
                "primary_score": result["primary_score"]
            })

    # rank: higher is better for all our primary scores (f1, r2)
    ranked = sorted(
        [row for row in leaderboard if "metrics" in row],
        key=lambda r: r["primary_score"],
        reverse=True,
    )
    failed = [row for row in leaderboard if "metrics" not in row]
    leaderboard = ranked + failed

    best_name = ranked[0]["model"] if ranked else None

    return problem_type, leaderboard, fitted, best_name, label_encoder


def feature_importance(pipe: Pipeline, X: pd.DataFrame) -> list:
    """Best-effort feature importance extraction for tree models / linear coefs."""
    try:
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        estimator = model.named_steps["estimator"] if isinstance(model, Pipeline) and "estimator" in model.named_steps else model
        feature_names = prep.get_feature_names_out()

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return []

        pairs = sorted(zip(feature_names, importances), key=lambda x: -abs(x[1]))[:15]
        return [{"feature": str(f), "importance": round(float(v), 4)} for f, v in pairs]
    except Exception:
        return []
