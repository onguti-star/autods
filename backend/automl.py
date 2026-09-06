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
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
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
LARGE_DATASET_MIN_SAMPLE = 30_000

# Decouples *which* model panel to use from *whether to sample*.  Below this
# row count the full candidate panel (all models, cross-validated ranking) is
# used; above it the "fast" panel — fewer models, all with built-in early
# stopping — kicks in so that medium-large datasets like 45k rows don't spend
# minutes on GradientBoosting / HistGradientBoosting.  This is intentionally
# lower than LARGE_DATASET_THRESHOLD (which controls *downsampling*) so a 45k
# dataset gets fast models *without* being downsampled.
FAST_MODELS_THRESHOLD = 30_000

# Model *selection* based on a single train/test split can crown a model that
# just got a lucky split rather than the one that actually generalizes best.
# For datasets small enough that it stays cheap, we additionally score every
# candidate with k-fold cross-validation and use that (much more stable)
# average score to pick the winner instead of the single-split score. This is
# skipped above this row count and for the already-time-boxed "large dataset"
# path so it can never be the thing that makes a run take too long.
CV_SELECTION_MAX_ROWS = 20_000
LARGE_DATASET_MAX_SAMPLE = 150_000
LARGE_DATASET_SAMPLE_FRACTION = 0.08


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
    one_hot_max_categories: int | None = None,
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
    if one_hot_max_categories is not None:
        # Hard cap on output columns per categorical feature. min_frequency alone
        # can still leave a wide matrix if a high-cardinality column (job titles,
        # addresses, zip codes...) has many categories that individually clear the
        # frequency bar. Capping the width here bounds how big the dense array
        # built downstream (see _to_dense) can get, which is what actually risks
        # exhausting memory -- and getting OS-killed -- on large datasets.
        categorical_kwargs["max_categories"] = one_hot_max_categories
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


def _compute_sample_weights(y_train):
    """Compute per-sample weights that up-weight the minority class.
    Used for models that do not natively support class_weight (e.g. GradientBoosting)."""
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    n_classes = len(classes)
    weight_map = {cls: total / (n_classes * cnt) for cls, cnt in zip(classes, counts)}
    return np.array([weight_map[c] for c in y_train])


def _cv_mean_score(name, model, X_train, y_train, problem_type, preprocessor, n_neighbors, n_rows, class_ratio=1.0):
    """Average k-fold cross-validation score for one candidate on the training
    split only (the held-out test set stays untouched for final reporting).

    The scoring metric deliberately mirrors the primary_score logic in
    _train_single_model so that CV ranking and single-split ranking agree on
    *which* model is best.  Previously this always used f1_weighted for
    classification while the ranking step used f1_macro for imbalanced datasets
    — causing the two to disagree and crown a different winner.

    Returns None (never raises) if CV isn't a good fit for this candidate or
    this dataset size -- callers fall back to the single-split score, so this
    is purely an enhancement, never a new failure mode."""
    if n_rows > CV_SELECTION_MAX_ROWS:
        return None
    try:
        candidate_model = clone(model)
        if name == "K-Nearest Neighbors":
            candidate_model.set_params(n_neighbors=n_neighbors)
        pipe = Pipeline(steps=[("prep", clone(preprocessor)), ("model", candidate_model)])

        if problem_type == "classification":
            # Enough folds to be meaningful, few enough to stay fast; never
            # more folds than the smallest class has members.
            min_class_count = int(pd.Series(y_train).value_counts().min())
            # Need at least 2 members per class to split; if any class is a
            # singleton we cannot do stratified CV safely — skip and fall back
            # to the single-split score rather than triggering sklearn warnings.
            if min_class_count < 2:
                return None
            n_splits = max(2, min(5, min_class_count))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            # Mirror the primary_score logic: prefer roc_auc for imbalanced
            # binary, f1_macro for imbalanced multiclass, f1_weighted otherwise.
            imbalanced = class_ratio > 3.0
            n_classes_train = len(np.unique(y_train))
            if imbalanced and n_classes_train == 2 and hasattr(candidate_model, "predict_proba"):
                scoring = "roc_auc"
            elif imbalanced:
                scoring = "f1_macro"
            else:
                scoring = "f1_weighted"
        else:
            n_splits = 5 if len(y_train) >= 50 else 3
            if len(y_train) < n_splits * 2:
                return None
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scoring = "r2"

        scores = cross_val_score(pipe, X_train, y_train, cv=splitter, scoring=scoring, n_jobs=1)
        return round(float(np.mean(scores)), 4)
    except Exception:
        return None


def _train_single_model(name, model, X_train, X_test, y_train, y_test, problem_type, preprocessor, n_neighbors, class_ratio=1.0):  # noqa: E501
    """Train a single model and return results.

    y_train is used alongside y_test to build the full label set for
    per_class_recall, so that rare classes absent from the test split still
    appear in the output (with recall=0.0) rather than being silently dropped.
    """
    try:
        candidate_model = clone(model)
        if name == "K-Nearest Neighbors":
            candidate_model.set_params(n_neighbors=n_neighbors)
        pipe = Pipeline(steps=[("prep", clone(preprocessor)), ("model", candidate_model)])

        # GradientBoostingClassifier does not accept class_weight; use sample_weight instead
        fit_params = {}
        if (
            problem_type == "classification"
            and class_ratio > 3.0
            and name == "Gradient Boosting"
        ):
            fit_params["model__sample_weight"] = _compute_sample_weights(y_train)

        pipe.fit(X_train, y_train, **fit_params)
        preds = pipe.predict(X_test)

        if problem_type == "classification":
            n_classes_test = len(np.unique(y_test))
            binary = n_classes_test == 2

            # ROC-AUC: use predict_proba if available, else skip
            roc_auc = None
            try:
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(X_test)
                    if binary:
                        from sklearn.metrics import roc_auc_score
                        roc_auc = round(float(roc_auc_score(y_test, proba[:, 1])), 4)
                    else:
                        from sklearn.metrics import roc_auc_score
                        roc_auc = round(float(roc_auc_score(
                            y_test, proba, multi_class="ovr", average="weighted"
                        )), 4)
            except Exception:
                roc_auc = None

            # Per-class recall (how many of each class did we catch).
            # np.unique(y_test) only contains classes that appear in the test
            # split — extremely rare classes may be absent entirely, making the
            # per_class_recall dict incomplete.  Using the full set of labels
            # seen during training (all unique encoded values across y_train +
            # y_test) ensures every class is represented, with 0.0 recall for
            # any that were too rare to land in the test split.
            from sklearn.metrics import classification_report
            all_labels = np.unique(np.concatenate([y_train, y_test]))
            report = classification_report(
                y_test, preds,
                labels=all_labels,
                output_dict=True,
                zero_division=0,
            )
            per_class_recall = {
                str(cls): round(float(report[str(cls)]["recall"]), 4)
                for cls in all_labels
                if str(cls) in report
            }

            metrics = {
                "accuracy":           round(float(accuracy_score(y_test, preds)), 4),
                "f1_weighted":        round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
                "f1_macro":           round(float(f1_score(y_test, preds, average="macro",    zero_division=0)), 4),
                "precision_weighted": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
                "recall_weighted":    round(float(recall_score(y_test, preds, average="weighted",    zero_division=0)), 4),
                "per_class_recall":   per_class_recall,
            }
            if roc_auc is not None:
                metrics["roc_auc"] = roc_auc

            # For imbalanced data roc_auc is the best ranking metric;
            # fall back to f1_macro (better than f1_weighted for imbalance),
            # then f1_weighted.
            if roc_auc is not None and class_ratio > 3.0:
                primary = roc_auc
            elif class_ratio > 3.0:
                primary = metrics["f1_macro"]
            else:
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


def _classification_models(n_classes: int, class_ratio: float = 1.0) -> dict:
    """Build classification models with imbalance-aware settings.

    class_ratio: majority_count / minority_count.  A ratio > 3 is considered
    imbalanced; models that support class weighting receive balanced weights so
    that minority classes (e.g. churned customers) are not drowned out.
    """
    imbalanced = class_ratio > 3.0
    cw = "balanced" if imbalanced else None

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=cw),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight=cw),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight=cw),
        # GradientBoostingClassifier does not support class_weight natively;
        # imbalance is handled via sample_weight inside _train_single_model.
        # It is also single-threaded (no n_jobs) and has no built-in early
        # stopping, so it is the slowest candidate on medium/large datasets.
        # Fewer trees with a shallower depth keeps it competitive while cutting
        # wall-clock time roughly in half.
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=50, max_depth=3),
        # early_stopping + n_iter_no_change bound the iteration count so the
        # full default 100-tree budget can't run when more trees don't help.
        "Histogram Gradient Boosting": _dense_model(HistGradientBoostingClassifier(
            random_state=42, class_weight=cw, early_stopping=True, n_iter_no_change=10,
        )),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }
    if XGBClassifier is not None:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        # scale_pos_weight tells XGBoost how much to up-weight the minority class
        spw = round(class_ratio, 2) if imbalanced and n_classes == 2 else 1
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective=objective,
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
        )
    return models


def _regression_models() -> dict:
    models = {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        # GradientBoostingRegressor is single-threaded (no n_jobs) and has no
        # built-in early stopping; the fewer-tree / shallower-depth settings
        # below keep it competitive while cutting wall-clock time roughly in
        # half.
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=50, max_depth=3),
        # early_stopping + n_iter_no_change bound the iteration count so the
        # full default 100-tree budget can't run when more trees don't help.
        "Histogram Gradient Boosting": _dense_model(HistGradientBoostingRegressor(
            random_state=42, early_stopping=True, n_iter_no_change=10,
        )),
        "K-Nearest Neighbors": KNeighborsRegressor(),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    return models


def _fast_classification_models(n_classes: int, class_ratio: float = 1.0) -> dict:
    """Fast models optimized for datasets above FAST_MODELS_THRESHOLD (30k rows), with imbalance awareness."""
    imbalanced = class_ratio > 3.0
    cw = "balanced" if imbalanced else None

    models = {
        "Linear Classifier (Fast)": SGDClassifier(
            loss="log_loss",
            max_iter=300,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            class_weight=cw,
            n_jobs=-1,
            random_state=42,
        ),
        # HistGradientBoosting is purpose-built for large N: it bins numeric
        # features once up front instead of scanning raw values per split,
        # so it stays fast at 100k+ rows while giving materially better
        # accuracy than a linear model or a very shallow XGBoost. It has its
        # own built-in early stopping, so runtime stays bounded even if the
        # sample size grows.
        "Histogram Gradient Boosting (Fast)": _dense_model(HistGradientBoostingClassifier(
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            class_weight=cw,
            random_state=42,
        )),
    }
    if XGBClassifier is not None:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        spw = round(class_ratio, 2) if imbalanced and n_classes == 2 else 1
        models["XGBoost (Fast)"] = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective=objective,
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return models


def _fast_regression_models() -> dict:
    """Fast models optimized for datasets above FAST_MODELS_THRESHOLD (30k rows)."""
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        # See note in _fast_classification_models: scales well to large N
        # with bounded runtime thanks to built-in early stopping.
        "Histogram Gradient Boosting (Fast)": _dense_model(HistGradientBoostingRegressor(
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )),
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

    # Imbalance ratio: used later to configure class-weighted models
    class_ratio = 1.0
    is_imbalanced = False

    label_encoder = None
    if problem_type == "classification":
        class_counts = y_raw.astype(str).value_counts()
        if len(class_counts) < 2:
            raise ValueError(f"Target column '{target}' has only one class. Choose a target with at least two values.")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        # Compute majority / minority ratio for imbalance handling
        majority = int(class_counts.iloc[0])
        minority = int(class_counts.iloc[-1])
        class_ratio = majority / max(minority, 1)
        is_imbalanced = class_ratio > 3.0
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
    # Use the fast model panel (fewer candidates, all with built-in early
    # stopping) once a dataset is large enough that the full panel — including
    # single-threaded GradientBoosting — would take minutes.  This is a
    # separate, lower threshold than LARGE_DATASET_THRESHOLD so 45k-row
    # datasets get fast models WITHOUT being downsampled.
    use_fast_models = original_n_rows > FAST_MODELS_THRESHOLD
    
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
        max_text_features=120 if use_fast_models else 300,
        # These caps used to apply only when the *row count* crossed
        # LARGE_DATASET_THRESHOLD, on the assumption that memory pressure
        # only comes from having many rows. But a single high-cardinality
        # categorical column (a raw ID, city, zip code, product code...) can
        # one-hot-encode into hundreds or thousands of columns regardless of
        # row count, and Histogram Gradient Boosting densifies that matrix
        # before training -- so a 45k-row dataset with one bad column can be
        # just as slow/memory-heavy as a true large dataset. Capping always,
        # not just above the row threshold, fixes that regardless of size.
        one_hot_min_frequency=10,
        one_hot_max_categories=50,
    )
    
    # OPTIMIZATION: Use faster models for large datasets -- fewer candidates,
    # all with built-in early stopping, so even medium-large datasets (e.g. 45k
    # rows) finish in seconds rather than minutes.
    if use_fast_models:
        if progress_callback:
            try:
                progress_callback(
                    f"Dataset has {original_n_rows:,} rows -- switching to fast model "
                    f"panel (early-stopping models, 3 candidates)..."
                )
            except Exception:
                pass
        candidates = (
            _fast_classification_models(len(np.unique(y)), class_ratio=class_ratio)
            if problem_type == "classification"
            else _fast_regression_models()
        )
    else:
        candidates = (
            _classification_models(len(np.unique(y)), class_ratio=class_ratio)
            if problem_type == "classification"
            else _regression_models()
        )

    # For imbalanced datasets notify via progress callback
    if is_imbalanced and progress_callback:
        try:
            progress_callback(
                f"Class imbalance detected (ratio {class_ratio:.1f}:1). "
                f"Applying balanced class weights to handle minority class…"
            )
        except Exception:
            pass

    n_neighbors = min(5, len(X_train))
    
    # OPTIMIZATION: Train models sequentially for large datasets to avoid memory issues
    # Parallel processing can cause memory overflow with large datasets
    results = []
    for name, model in candidates.items():
        if progress_callback:
            try:
                # Include memory snapshot per model so it's visible in /diag
                # even if the process is killed between models.
                try:
                    import psutil, os as _os, time as _time
                    proc = psutil.Process(_os.getpid())
                    rss_mb = proc.memory_info().rss / 1_048_576
                    mem_note = f" | mem: {rss_mb:.0f} MB"
                except Exception:
                    mem_note = ""
                    _time = __import__("time")
                progress_callback(f"Training {name}...{mem_note}")
            except Exception:
                pass  # never let a progress-reporting hiccup break training
        result = _train_single_model(
            name, model, X_train, X_test, y_train, y_test, problem_type, preprocessor, n_neighbors,
            class_ratio=class_ratio,
        )
        if "error" not in result and not use_fast_models:
            result["cv_score"] = _cv_mean_score(
                name, model, X_train, y_train, problem_type, preprocessor, n_neighbors, n_rows,
                class_ratio=class_ratio,
            )
        results.append(result)

    fitted = {}
    leaderboard = []
    
    for result in results:
        if "error" in result:
            leaderboard.append({"model": result["model"], "error": result["error"]})
        else:
            fitted[result["model"]] = result["fitted"]
            row = {
                "model": result["model"],
                "metrics": result["metrics"],
                "primary_score": result["primary_score"],
            }
            if result.get("cv_score") is not None:
                row["cv_score"] = result["cv_score"]
            leaderboard.append(row)

    # Rank by cross-validated score when we have it for every surviving
    # candidate (much more stable than a single train/test split -- see
    # _cv_mean_score). Otherwise fall back to the single-split primary_score,
    # e.g. on large datasets where CV is skipped to keep runtime bounded.
    scored_rows = [row for row in leaderboard if "metrics" in row]
    use_cv_rank = len(scored_rows) > 0 and all("cv_score" in row for row in scored_rows)
    rank_key = (lambda r: r["cv_score"]) if use_cv_rank else (lambda r: r["primary_score"])

    ranked = sorted(scored_rows, key=rank_key, reverse=True)
    failed = [row for row in leaderboard if "metrics" not in row]
    leaderboard = ranked + failed

    best_name = ranked[0]["model"] if ranked else None

    return problem_type, leaderboard, fitted, best_name, label_encoder


def feature_importance(pipe: Pipeline, X: pd.DataFrame = None) -> list:
    """Best-effort feature importance extraction for tree models / linear coefs.

    X is accepted for backward compatibility but is not used -- feature names
    come from the already-fitted preprocessor and importances from the already
    -fitted estimator, so callers no longer need to hand over a (possibly huge)
    copy of the training frame just to call this."""
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