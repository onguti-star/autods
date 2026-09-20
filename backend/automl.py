"""
AutoML core: given a dataframe + target column, this:
  1. Infers the problem type (classification vs regression)
  2. Builds a preprocessing pipeline (impute, scale, one-hot encode, target encode)
  3. Trains a panel of candidate models (including LightGBM, CatBoost, Stacking)
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
    StackingClassifier,
    StackingRegressor,
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
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.decomposition import PCA

from . import nlp

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # XGBoost is optional; scikit-learn models remain the default fallback.
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None


LARGE_DATASET_THRESHOLD = 100_000
LARGE_DATASET_MIN_SAMPLE = 30_000

# Decouples *which* model panel to use from *whether to sample*.  Below this
# row count the full candidate panel (all models, cross-validated ranking) is
# used; above it the "fast" panel kicks in so that large datasets don't spend
# minutes on single-threaded GradientBoosting.  Raised from 30k to 50k, and
# the fast panel itself was made less lossy (see _fast_classification_models
# / _fast_regression_models below) -- it now keeps a tree-ensemble candidate
# alongside the linear/HGB ones instead of relying on a single fast estimator
# per model family, so datasets in the 30k-50k range that used to lose real
# accuracy to the fast switch now get a fuller panel instead.
FAST_MODELS_THRESHOLD = 50_000

# Model *selection* based on a single train/test split can crown a model that
# just got a lucky split rather than the one that actually generalizes best.
# For datasets small enough that it stays cheap, we additionally score every
# candidate with k-fold cross-validation and use that (much more stable)
# average score to pick the winner instead of the single-split score. Raised
# from 20k to 30k to match FAST_MODELS_THRESHOLD's old boundary, so CV-based
# selection -- the biggest lever on "did we actually pick the best model" --
# covers the same range the full panel now runs on. Still skipped above this
# and for the already time-boxed "large dataset" path so it can't be the
# thing that makes a run take too long.
CV_SELECTION_MAX_ROWS = 30_000
# Sample size/fraction used once a dataset is large enough to be downsampled
# at all (> LARGE_DATASET_THRESHOLD). Raised the fraction and the cap so large
# datasets keep materially more rows -- more training data is usually a
# bigger accuracy lever than which fast model gets picked -- while the
# early-stopping fast models keep the extra rows from blowing up runtime.
LARGE_DATASET_MAX_SAMPLE = 250_000
LARGE_DATASET_SAMPLE_FRACTION = 0.15


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
    y: pd.Series = None,
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
    
    # Check cardinality to apply Target Encoding for high-cardinality columns
    high_card_cols = []
    low_card_cols = []
    for col in categorical_cols:
        if X[col].nunique() > 15:
            high_card_cols.append(col)
        else:
            low_card_cols.append(col)

    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**categorical_kwargs)),
    ])

    high_card_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("target_enc", TargetEncoder(target_type="continuous" if y is not None and pd.api.types.is_numeric_dtype(y) else "binary")),
        ("scale", StandardScaler())
    ])

    transformers = [
        ("num", numeric_pipe, numeric_cols),
    ]
    if low_card_cols:
        transformers.append(("cat_low", categorical_pipe, low_card_cols))
    if high_card_cols:
        transformers.append(("cat_high", high_card_pipe, high_card_cols))

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
    if n_rows > CV_SELECTION_MAX_ROWS or "Stacking" in name:
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


def _score_classification(pipe, X_test, y_test, y_train, class_ratio=1.0):
    """Compute the classification metrics dict + primary (ranking) score for
    an already-fitted pipeline. Factored out of _train_single_model so
    _refine_winner (hyperparameter tuning) can score a re-fitted pipeline
    with exactly the same logic instead of duplicating it.

    y_train is used alongside y_test to build the full label set for
    per_class_recall, so that rare classes absent from the test split still
    appear in the output (with recall=0.0) rather than being silently dropped.
    """
    preds = pipe.predict(X_test)
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
    return metrics, primary


def _score_regression(pipe, X_test, y_test):
    """Regression counterpart to _score_classification -- see that
    docstring for why this is factored out."""
    preds = pipe.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, preds))
    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
    }
    return metrics, metrics["r2"]


def _train_single_model(name, model, X_train, X_test, y_train, y_test, problem_type, preprocessor, n_neighbors, class_ratio=1.0):  # noqa: E501
    """Train a single model and return results."""
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

        if problem_type == "classification":
            metrics, primary = _score_classification(pipe, X_test, y_test, y_train, class_ratio)
        else:
            metrics, primary = _score_regression(pipe, X_test, y_test)

        return {"model": name, "metrics": metrics, "primary_score": primary, "fitted": pipe}
    except Exception as e:
        return {"model": name, "error": str(e)}


# Light hyperparameter search grids for the winning model, keyed by the same
# display names used in the candidate panels above. Only tree/boosting models
# get a grid -- these are the models where a handful of extra trees, a bit
# more depth, or a different learning rate reliably buys accuracy; linear
# models and KNN have too little to tune to be worth the extra fit time.
# Params are written without the "model__" (or "model__estimator__" for
# models wrapped by _dense_model, i.e. HistGradientBoosting) prefix --
# _refine_winner adds the right prefix based on the pipeline shape.
_TUNE_PARAM_GRIDS = {
    "Random Forest":       {"n_estimators": [200, 300, 400], "max_depth": [8, 14, 20, None], "min_samples_leaf": [1, 2, 4]},
    "Extra Trees":         {"n_estimators": [200, 300, 400], "max_depth": [10, 16, None], "min_samples_leaf": [1, 2, 5]},
    "Extra Trees (Fast)":  {"n_estimators": [100, 150, 200], "max_depth": [10, 14, 20], "min_samples_leaf": [2, 5, 10]},
    "Gradient Boosting":   {"n_estimators": [80, 120, 160], "max_depth": [2, 3, 4, 5], "learning_rate": [0.03, 0.05, 0.1, 0.2]},
    "LightGBM":            {"n_estimators": [100, 150, 250], "num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1]},
    "CatBoost":            {"iterations": [150, 250, 350], "depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1]},
    "XGBoost":             {"n_estimators": [80, 100, 150], "max_depth": [3, 4, 5, 6], "learning_rate": [0.03, 0.05, 0.1, 0.2], "subsample": [0.7, 0.8, 0.9, 1.0], "colsample_bytree": [0.7, 0.8, 0.9, 1.0]},
    "XGBoost (Fast)":      {"n_estimators": [60, 80, 100], "max_depth": [3, 4, 5], "learning_rate": [0.05, 0.1, 0.2]},
    "Histogram Gradient Boosting":        {"max_depth": [None, 6, 10], "learning_rate": [0.03, 0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63]},
    "Histogram Gradient Boosting (Fast)": {"max_depth": [None, 6, 10], "learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31]},
}


def _refine_winner(
    name, model_template, X_train, X_test, y_train, y_test, problem_type,
    preprocessor, n_rows, class_ratio=1.0, progress_callback=None,
):
    """Randomized hyperparameter search around the winning model's defaults.

    Model *selection* (which algorithm) and CV ranking only ever compare
    fixed, hand-picked hyperparameters -- they can tell you Random Forest
    beat Ridge, but never that a slightly deeper Random Forest would have
    beaten both by more. This closes that gap for the single model that
    actually gets used, without paying the search cost across every
    candidate in the panel.

    Guarded by CV_SELECTION_MAX_ROWS (same budget used for CV-based
    selection) so it can't be the thing that makes a run take too long, and
    by _TUNE_PARAM_GRIDS so it only runs for models where tuning reliably
    pays off. Returns None (never raises) on any failure or when refinement
    doesn't apply -- callers keep the original fitted model in that case.
    """
    grid = _TUNE_PARAM_GRIDS.get(name)
    if grid is None or n_rows > CV_SELECTION_MAX_ROWS:
        return None
    try:
        from sklearn.model_selection import RandomizedSearchCV

        candidate_model = clone(model_template)
        pipe = Pipeline(steps=[("prep", clone(preprocessor)), ("model", candidate_model)])
        # _dense_model wraps the real estimator in a nested Pipeline (steps
        # "dense", "estimator") so sparse one-hot/TF-IDF output can be
        # densified before models like HistGradientBoosting that need dense
        # input -- see _dense_model / _to_dense above. That nesting changes
        # which prefix RandomizedSearchCV needs to reach the estimator's params.
        is_wrapped = isinstance(candidate_model, Pipeline)
        prefix = "model__estimator__" if is_wrapped else "model__"
        param_distributions = {f"{prefix}{k}": v for k, v in grid.items()}

        if problem_type == "classification":
            min_class_count = int(pd.Series(y_train).value_counts().min())
            if min_class_count < 2:
                return None
            n_splits = max(2, min(3, min_class_count))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            imbalanced = class_ratio > 3.0
            n_classes_train = len(np.unique(y_train))
            if imbalanced and n_classes_train == 2 and hasattr(candidate_model, "predict_proba"):
                scoring = "roc_auc"
            elif imbalanced:
                scoring = "f1_macro"
            else:
                scoring = "f1_weighted"
        else:
            n_splits = 3
            if len(y_train) < n_splits * 2:
                return None
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scoring = "r2"

        if progress_callback:
            try:
                progress_callback(f"Fine-tuning {name} hyperparameters...")
            except Exception:
                pass

        search = RandomizedSearchCV(
            pipe,
            param_distributions,
            n_iter=8,
            cv=splitter,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            error_score=np.nan,
        )
        # Mirror _train_single_model's imbalance handling for Gradient
        # Boosting (no native class_weight support) so tuning doesn't
        # regress imbalanced-data accuracy relative to the untuned candidate.
        fit_params = {}
        if problem_type == "classification" and class_ratio > 3.0 and name == "Gradient Boosting":
            fit_params["model__sample_weight"] = _compute_sample_weights(y_train)
        search.fit(X_train, y_train, **fit_params)
        tuned_pipe = search.best_estimator_

        if problem_type == "classification":
            metrics, primary = _score_classification(tuned_pipe, X_test, y_test, y_train, class_ratio)
        else:
            metrics, primary = _score_regression(tuned_pipe, X_test, y_test)

        return {"metrics": metrics, "primary_score": primary, "fitted": tuned_pipe}
    except Exception:
        return None


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
        # This panel now only runs below FAST_MODELS_THRESHOLD (50k rows), so
        # there's headroom to give it a fuller tree budget than before (was
        # 50/depth-3) without the runtime getting out of hand.
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=120, max_depth=4),
        # n_iter_no_change raised from 10 to 20: HistGB stops as soon as
        # validation score hasn't improved for that many iterations, so a
        # low value can cut training short on plateaus that would still
        # improve with a bit more patience. Costs a bit more time per model,
        # not per-tree cost, so it stays cheap.
        "Histogram Gradient Boosting": _dense_model(HistGradientBoostingClassifier(
            random_state=42, class_weight=cw, early_stopping=True, n_iter_no_change=20,
        )),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            class_weight=cw,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
        )
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
        # built-in early stopping. This panel only runs below
        # FAST_MODELS_THRESHOLD (50k rows) now, so there's room for a fuller
        # tree budget than before (was 50/depth-3).
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=120, max_depth=4),
        # n_iter_no_change raised from 10 to 20 for the same reason as the
        # classifier variant -- more patience before declaring a plateau.
        "Histogram Gradient Boosting": _dense_model(HistGradientBoostingRegressor(
            random_state=42, early_stopping=True, n_iter_no_change=20,
        )),
        "K-Nearest Neighbors": KNeighborsRegressor(),
    }
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
        )
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
        # accuracy than a linear model or a very shallow XGBoost. max_iter and
        # n_iter_no_change both raised (150->200, 10->15) for more headroom
        # before early stopping kicks in -- it still has its own built-in
        # early stopping, so runtime stays bounded even if the sample size
        # grows.
        "Histogram Gradient Boosting (Fast)": _dense_model(HistGradientBoostingClassifier(
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            class_weight=cw,
            random_state=42,
        )),
        # The previous fast panel had no bagged/randomized-split tree
        # candidate at all -- just one linear model and one boosting model.
        # ExtraTrees builds each split from a random threshold rather than
        # searching for the best one, which makes it considerably cheaper
        # than RandomForest per tree, so a capped depth/estimator count stays
        # fast even at 100k+ rows (and parallelizes via n_jobs=-1) while
        # adding real model diversity -- it often catches different signal
        # than boosting, which is worth having in the leaderboard comparison.
        "Extra Trees (Fast)": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
            class_weight=cw,
        ),
    }
    if LGBMClassifier is not None:
        models["LightGBM (Fast)"] = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            class_weight=cw,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if XGBClassifier is not None:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        spw = round(class_ratio, 2) if imbalanced and n_classes == 2 else 1
        # n_estimators raised 50->80 and max_depth 3->4: XGBoost's histogram
        # tree method (tree_method="hist") is what makes it fast on large N,
        # not the shallow budget, so there's room to give it more capacity
        # without materially hurting runtime.
        models["XGBoost (Fast)"] = XGBClassifier(
            n_estimators=80,
            max_depth=4,
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
        # with bounded runtime thanks to built-in early stopping. Budget
        # raised the same way (max_iter 200->250, n_iter_no_change 10->15).
        "Histogram Gradient Boosting (Fast)": _dense_model(HistGradientBoostingRegressor(
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
        )),
        # See note in _fast_classification_models: adds a bagged/randomized
        # tree candidate the fast panel previously lacked entirely.
        "Extra Trees (Fast)": ExtraTreesRegressor(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
    }
    if LGBMRegressor is not None:
        models["LightGBM (Fast)"] = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if XGBRegressor is not None:
        models["XGBoost (Fast)"] = XGBRegressor(
            n_estimators=80,
            max_depth=4,
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
) -> Tuple[str, list, dict, str, object, list]:
    """
    Returns: (problem_type, leaderboard, fitted_pipelines_by_name, best_name,
              label_encoder_or_None, best_model_feature_importance)
    
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
        y=y_train,
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

    # --- Stacking Ensemble Step ---
    # Take top 3 successful non-ensemble models and fit a meta-learner stack
    successful_results = [r for r in results if "error" not in r]
    if len(successful_results) >= 2:
        top_base = sorted(successful_results, key=lambda r: r["primary_score"], reverse=True)[:3]
        base_estimators = [(r["model"], r["fitted"]) for r in top_base]
        
        try:
            if progress_callback:
                progress_callback("Building Stacking Ensemble from top base models...")
                
            if problem_type == "classification":
                cw = "balanced" if class_ratio > 3.0 else None
                stack_estimator = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(class_weight=cw),
                    cv=3,
                    n_jobs=-1
                )
            else:
                stack_estimator = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=Ridge(),
                    cv=3,
                    n_jobs=-1
                )
            
            stack_estimator.fit(X_train, y_train)
            
            if problem_type == "classification":
                metrics, primary = _score_classification(stack_estimator, X_test, y_test, y_train, class_ratio)
            else:
                metrics, primary = _score_regression(stack_estimator, X_test, y_test)
                
            results.append({
                "model": "Stacking Ensemble",
                "metrics": metrics,
                "primary_score": primary,
                "fitted": stack_estimator
            })
        except Exception as e:
            results.append({"model": "Stacking Ensemble", "error": str(e)})

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

    # Hyperparameter refinement: fixed-hyperparameter candidates chose *which*
    # model wins, but never tried varying that model's own settings. Search a
    # small grid around the winner and keep the tuned version only if it
    # actually scores better on the held-out test set -- this can never make
    # the leaderboard worse, only better. See _refine_winner for the budget
    # guard that keeps this from slowing down large-dataset runs.
    if best_name is not None and best_name in candidates:
        refined = _refine_winner(
            best_name, candidates[best_name], X_train, X_test, y_train, y_test,
            problem_type, preprocessor, n_rows, class_ratio=class_ratio,
            progress_callback=progress_callback,
        )
        if refined is not None and refined["primary_score"] > ranked[0]["primary_score"]:
            fitted[best_name] = refined["fitted"]
            ranked[0]["metrics"] = refined["metrics"]
            ranked[0]["primary_score"] = refined["primary_score"]
            ranked[0]["tuned"] = True
            leaderboard = ranked + failed

    # Compute feature importance for the winning model right here, while
    # X_test/y_test (the raw, un-preprocessed held-out split) are still in
    # scope -- this is what lets the permutation-importance fallback above
    # kick in for models like HistGradientBoosting that don't expose
    # feature_importances_/coef_. Callers used to recompute this later from
    # just the fitted pipeline with no data at all, which meant the fallback
    # never had anything to permute and silently returned "not available".
    best_importance = (
        feature_importance(fitted[best_name], X_test, y_test) if best_name else []
    )

    return problem_type, leaderboard, fitted, best_name, label_encoder, best_importance


def feature_importance(pipe: Pipeline, X: pd.DataFrame = None, y=None) -> list:
    """Best-effort feature importance extraction for tree models / linear coefs.

    X/y are optional. When the fitted estimator exposes neither
    `feature_importances_` (tree ensembles like RandomForest/ExtraTrees) nor
    `coef_` (linear models) -- which is the case for HistGradientBoosting,
    scikit-learn's own fast-model default -- there is no importance baked
    into the fitted object at all. In that case, if labelled data (X, y) is
    supplied, we fall back to permutation importance computed directly on
    the full pipeline: shuffle each raw input column in turn and measure how
    much the held-out score drops. This is model-agnostic, so it works for
    HistGradientBoosting (and anything else) the same way, and it's computed
    on the ORIGINAL columns (not the expanded one-hot/TF-IDF ones), which is
    also more readable for the chart."""
    try:
        # If it's a Stacking Ensenble, extracting direct importance is difficult.
        # We fall back to permutation importance on the whole pipeline if X/y are available.
        if hasattr(pipe, "named_steps") and "prep" in pipe.named_steps:
            prep = pipe.named_steps["prep"]
            model = pipe.named_steps["model"]
            estimator = model.named_steps["estimator"] if isinstance(model, Pipeline) and "estimator" in model.named_steps else model
            feature_names = prep.get_feature_names_out()

            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                coef = estimator.coef_
                importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
            elif X is not None and y is not None and len(X) > 0:
                from sklearn.inspection import permutation_importance

                sample_X, sample_y = X, y
                # Cap the sample so this stays fast on large held-out splits --
                # permutation importance refits nothing but re-predicts the full
                # pipeline once per column per repeat, which adds up.
                if len(sample_X) > 2000:
                    idx = np.random.RandomState(42).choice(len(sample_X), 2000, replace=False)
                    sample_X = sample_X.iloc[idx] if hasattr(sample_X, "iloc") else sample_X[idx]
                    sample_y = sample_y[idx] if not hasattr(sample_y, "iloc") else sample_y.iloc[idx]
                result = permutation_importance(
                    pipe, sample_X, sample_y, n_repeats=5, random_state=42, n_jobs=1
                )
                importances = result.importances_mean
                # Permutation importance is measured against the raw input
                # columns (it shuffles them before they hit the preprocessor),
                # not the post-encoding feature names used in the branches above.
                feature_names = list(sample_X.columns)
            else:
                return []

            pairs = sorted(zip(feature_names, importances), key=lambda x: -abs(x[1]))[:15]
            # Permutation importance can come out slightly negative for pure-noise
            # columns (shuffling them occasionally helps by chance). Drop those so
            # the chart doesn't show a feature that provably doesn't matter --
            # but if every single one is <=0 (degenerate/near-constant model),
            # keep the top few anyway rather than showing nothing.
            positive = [(f, v) for f, v in pairs if v > 0]
            pairs = positive if positive else pairs[:5]
            return [{"feature": str(f), "importance": round(float(v), 4)} for f, v in pairs]
        
        # Fallback block specifically for handling the new Stacking Ensembles
        if X is not None and y is not None and len(X) > 0:
            from sklearn.inspection import permutation_importance
            sample_X = X.iloc[:1000] if len(X) > 1000 else X
            sample_y = y[:1000] if len(y) > 1000 else y
            res = permutation_importance(pipe, sample_X, sample_y, n_repeats=3, random_state=42)
            pairs = sorted(zip(sample_X.columns, res.importances_mean), key=lambda x: -abs(x[1]))[:15]
            return [{"feature": str(f), "importance": round(float(v), 4)} for f, v in pairs if v > 0]

        return []
    except Exception:
        return []