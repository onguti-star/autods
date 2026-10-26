"""
Recommender systems module: works on *any* dataset, not just Netflix/movies.

Three modes, auto-detected from the dataframe's shape:

  1. "collaborative"  — long format with (user, item, rating) columns,
                         e.g. userId, movieId, rating.
                         Uses item-based kNN (cosine similarity) and
                         matrix-factorization (Truncated SVD) to predict
                         ratings / recommend unseen items to a user.

  2. "content"        — one row per item with descriptive feature columns
                         (genres, text, category, tags, numeric attributes...).
                         Uses TF-IDF (for text/categoricals) + numeric
                         similarity to recommend items similar to a given item.

  3. "implicit"       — long format with just (user, item) interactions
                         (clicks, purchases, watches — no explicit rating).
                         Treated as collaborative filtering with implicit
                         feedback (binary / count matrix).

All functions are self-contained and return JSON-serializable results,
mirroring the style of unsupervised.py / automl.py in this codebase.
"""
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

warnings.filterwarnings("ignore")


# ── Detection ─────────────────────────────────────────────────────────────

def _looks_like_id_column(series: pd.Series) -> bool:
    """High-cardinality, mostly-unique-ish column -> likely a user/item id."""
    n = len(series)
    if n == 0:
        return False
    nunique = series.nunique(dropna=True)
    return nunique >= 2 and nunique <= n and (nunique / n) > 0.01


def _looks_like_rating_column(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = series.dropna()
    if vals.empty:
        return False
    nunique = vals.nunique()
    lo, hi = vals.min(), vals.max()
    # small set of distinct numeric values in a plausible rating-scale range
    return nunique <= 20 and lo >= 0 and hi <= 100


def _as_plain_str_series(series: pd.Series, fill: str = "") -> pd.Series:
    """Safely coerce to plain string, filling missing values.

    A Categorical column (common after AutoDS's own type inference/cleaning)
    rejects .fillna() with any value that isn't already one of its declared
    categories, so the dtype has to be dropped to plain object first.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    return series.fillna(fill).astype(str)


def _sample_str_values(series: pd.Series, n: int = 50) -> list[str]:
    vals = series.dropna().astype(str)
    if len(vals) > n:
        vals = vals.sample(n, random_state=0)
    return vals.tolist()


def _looks_like_text_column(series: pd.Series) -> bool:
    """Free-flowing prose (a description/overview/review) -> good TF-IDF input."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    vals = _sample_str_values(series)
    if not vals:
        return False
    avg_words = sum(len(v.split()) for v in vals) / len(vals)
    return avg_words >= 6


def _looks_like_list_column(series: pd.Series) -> bool:
    """Short delimited tags (genres, categories: 'Action|Comedy' or 'Action, Comedy')."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    vals = _sample_str_values(series)
    if not vals:
        return False
    delimited = [v for v in vals if "|" in v or ("," in v and len(v) < 120)]
    if len(delimited) / len(vals) < 0.5:
        return False
    avg_tokens = sum(len(v.replace("|", ",").split(",")) for v in delimited) / len(delimited)
    avg_token_len = sum(len(t.strip()) for v in delimited for t in v.replace("|", ",").split(",")) / max(
        sum(len(v.replace("|", ",").split(",")) for v in delimited), 1
    )
    return 1.5 <= avg_tokens <= 8 and avg_token_len <= 30


def detect_recommender_setup(df: pd.DataFrame) -> dict:
    """
    Inspect a dataframe and suggest how to build a recommender from it.
    Returns a dict describing the best-guess mode and candidate columns,
    so the UI can let the user confirm/adjust before running.
    """
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    def find_by_keywords(keywords):
        for c in cols:
            lc = lower[c]
            if any(k in lc for k in keywords):
                return c
        return None

    n = len(df)
    title_like_col = find_by_keywords(["title", "name"])

    user_col = find_by_keywords(["user_id", "userid", "user", "customer", "member", "visitor", "reviewer"])
    item_col = find_by_keywords(["item_id", "itemid", "item", "movie", "product", "book", "track", "song", "sku", "asin"])
    rating_col = find_by_keywords(["rating", "score", "stars", "rank"])

    def repeats(col):
        """True if this column's values repeat (a real foreign key in an
        interaction log), not just a row-unique identifier/title."""
        return col is not None and n > 0 and df[col].nunique(dropna=True) < n

    # A one-row-per-item table (e.g. a movie catalogue with title/genres/
    # description) should never be mistaken for a user/item interaction log,
    # even if a generic "id-like" column heuristic would otherwise match.
    # Require BOTH columns to actually repeat across rows before trusting
    # name-based matches, and fall back to structural detection only among
    # columns that repeat.
    if not (repeats(user_col) and repeats(item_col) and user_col != item_col):
        user_col, item_col = None, None
        candidate_ids = [
            c for c in cols
            if c != rating_col and c != title_like_col and _looks_like_id_column(df[c]) and repeats(c)
        ]
        if len(candidate_ids) >= 1:
            user_col = candidate_ids[0]
        if len(candidate_ids) >= 2:
            item_col = next((c for c in candidate_ids if c != user_col), None)

    # A keyword match on "rating" is only trustworthy as an explicit-rating
    # column if it's actually numeric (e.g. 1-5 stars) -- a maturity rating
    # like "PG-13"/"TV-MA" matches the word "rating" but is a categorical
    # attribute, not something to feed into matrix factorization.
    if rating_col is not None and not _looks_like_rating_column(df[rating_col]):
        rating_col = None
    if rating_col is None:
        for c in cols:
            if c not in (user_col, item_col) and _looks_like_rating_column(df[c]):
                rating_col = c
                break

    # A one-row-per-item catalogue (title/genres/director/description...)
    # almost always has *other* repeating categorical columns (country,
    # director, listed_in...) that structurally resemble a user/item foreign
    # key pair even though they aren't one. If there's a title/name column
    # that's unique (or nearly so) per row, this is a content catalogue, not
    # an interaction log -- a genuine ratings/interaction log would instead
    # have the same item appear across many rows (one per user), so no
    # column would be anywhere near row-unique.
    is_item_catalog = title_like_col is not None and df[title_like_col].nunique(dropna=True) / max(n, 1) > 0.8

    has_user_item = (
        not is_item_catalog
        and user_col is not None and item_col is not None and user_col != item_col
        and repeats(user_col) and repeats(item_col)
        and n >= 4  # a handful of rows isn't a real interaction log
    )

    # content-based candidate: one row per item, with text/categorical/numeric feature columns
    item_like_col = title_like_col or (item_col if not has_user_item else None)
    feature_cols, suggested_text_cols, suggested_list_cols = [], [], []
    if item_like_col:
        is_mostly_unique = df[item_like_col].nunique() / max(n, 1) > 0.8
        if is_mostly_unique:
            for c in cols:
                if c == item_like_col:
                    continue
                is_text = _looks_like_text_column(df[c])
                is_list = (not is_text) and _looks_like_list_column(df[c])
                near_unique = df[c].nunique(dropna=True) / max(n, 1) > 0.8
                # Drop other near-row-unique columns (raw ids, dates) --
                # they carry no similarity signal and fit_content() would
                # silently skip them as high-cardinality categoricals anyway.
                # Free text is the one near-unique case worth keeping: that's
                # exactly where TF-IDF similarity does real work.
                if near_unique and not pd.api.types.is_numeric_dtype(df[c]) and not is_text:
                    continue
                feature_cols.append(c)
                if is_text:
                    suggested_text_cols.append(c)
                elif is_list:
                    suggested_list_cols.append(c)

    # Content goes first when this is clearly an item catalogue, so it wins
    # as the best guess even if a (now-suppressed) user/item pattern also
    # happened to match structurally.
    modes = []
    if is_item_catalog and item_like_col and feature_cols:
        modes.append({
            "mode": "content",
            "description": "One row per item with descriptive attributes — build a content-based 'more like this' recommender using item similarity.",
            "item_col": item_like_col,
            "feature_cols": feature_cols,
            "text_cols": suggested_text_cols,
            "list_cols": suggested_list_cols,
        })
    if has_user_item and rating_col:
        modes.append({
            "mode": "collaborative",
            "description": "Explicit ratings found — build a user/item collaborative-filtering recommender (item-based kNN + matrix factorization), just like the Netflix prize setup.",
            "user_col": user_col,
            "item_col": item_col,
            "rating_col": rating_col,
        })
    if has_user_item and not rating_col:
        modes.append({
            "mode": "implicit",
            "description": "User/item interactions found with no rating column — build an implicit-feedback recommender (clicks/purchases/watches treated as positive signal).",
            "user_col": user_col,
            "item_col": item_col,
        })
    if item_like_col and feature_cols and not is_item_catalog:
        modes.append({
            "mode": "content",
            "description": "One row per item with descriptive attributes — build a content-based 'more like this' recommender using item similarity.",
            "item_col": item_like_col,
            "feature_cols": feature_cols,
            "text_cols": suggested_text_cols,
            "list_cols": suggested_list_cols,
        })

    best = modes[0] if modes else None
    return {
        "ok": best is not None,
        "best_guess": best,
        "all_options": modes,
        "message": (
            "Could not confidently detect a user/item or item-feature structure. "
            "You can still pick columns manually."
            if best is None else
            f"Detected a good fit for '{best['mode']}' recommendations."
        ),
    }


# ── Shared helpers ───────────────────────────────────────────────────────

def _safe_json(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 6)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def _build_interaction_matrix(df: pd.DataFrame, user_col: str, item_col: str, rating_col: Optional[str]):
    """Build a sparse user x item matrix (ratings if given, else binary 1s)."""
    work = df[[user_col, item_col] + ([rating_col] if rating_col else [])].dropna(subset=[user_col, item_col])
    users = work[user_col].astype(str)
    items = work[item_col].astype(str)

    user_ids = pd.Index(sorted(users.unique()))
    item_ids = pd.Index(sorted(items.unique()))
    user_idx = users.map({u: i for i, u in enumerate(user_ids)})
    item_idx = items.map({it: i for i, it in enumerate(item_ids)})

    if rating_col:
        values = work[rating_col].astype(float).fillna(work[rating_col].astype(float).mean())
    else:
        values = np.ones(len(work))

    mat = csr_matrix((values, (user_idx, item_idx)), shape=(len(user_ids), len(item_ids)))
    return mat, user_ids, item_ids


# ── Collaborative / implicit filtering ──────────────────────────────────

def fit_collaborative(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: Optional[str] = None,
    n_components: int = 20,
) -> dict:
    """
    Fit an item-based kNN similarity model + a matrix-factorization (SVD)
    model over the user-item interaction matrix. Returns a compact,
    JSON-serializable model snapshot that recommend_for_user() re-derives from.
    """
    mat, user_ids, item_ids = _build_interaction_matrix(df, user_col, item_col, rating_col)
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        raise ValueError("Need at least 2 distinct users and 2 distinct items to build a collaborative recommender.")

    n_components = int(min(n_components, max(2, min(mat.shape) - 1)))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(mat)          # users x k
    item_factors = svd.components_.T                # items x k

    item_sim = cosine_similarity(mat.T)  # items x items, from raw interactions

    return {
        "mode": "collaborative" if rating_col else "implicit",
        "user_col": user_col,
        "item_col": item_col,
        "rating_col": rating_col,
        "n_users": int(mat.shape[0]),
        "n_items": int(mat.shape[1]),
        "n_components": n_components,
        "sparsity": round(1 - mat.nnz / (mat.shape[0] * mat.shape[1]), 4),
        "explained_variance_ratio": round(float(svd.explained_variance_ratio_.sum()), 4),
        "_user_ids": user_ids,
        "_item_ids": item_ids,
        "_interaction_matrix": mat,
        "_user_factors": user_factors,
        "_item_factors": item_factors,
        "_item_similarity": item_sim,
    }


def recommend_for_user(model: dict, user_id: Any, top_n: int = 10, method: str = "hybrid") -> dict:
    """
    Recommend items for a given user using a model from fit_collaborative().
    method: "svd" (matrix factorization), "item_knn" (item-based similarity
    over the user's history), or "hybrid" (average of both, default).
    """
    user_ids = model["_user_ids"]
    item_ids = model["_item_ids"]
    mat = model["_interaction_matrix"]
    key = str(user_id)
    if key not in set(user_ids):
        raise ValueError(f"Unknown user '{user_id}'. This user has no interaction history in the training data.")

    u = int(np.where(user_ids == key)[0][0])
    seen_mask = mat[u].toarray().ravel() > 0
    seen_items = set(item_ids[seen_mask])

    scores = np.zeros(len(item_ids))

    if method in ("svd", "hybrid"):
        user_vec = model["_user_factors"][u]
        svd_scores = model["_item_factors"] @ user_vec
        svd_scores = (svd_scores - svd_scores.min()) / (np.ptp(svd_scores) + 1e-9)
        scores = scores + svd_scores

    if method in ("item_knn", "hybrid"):
        user_row = mat[u].toarray().ravel()
        sim = model["_item_similarity"]
        knn_scores = sim @ user_row
        knn_scores = (knn_scores - knn_scores.min()) / (np.ptp(knn_scores) + 1e-9)
        scores = scores + knn_scores

    ranked = np.argsort(-scores)
    recs = []
    for idx in ranked:
        item = item_ids[idx]
        if item in seen_items:
            continue
        recs.append({"item": str(item), "score": round(float(scores[idx]), 4)})
        if len(recs) >= top_n:
            break

    return {
        "user": key,
        "method": method,
        "n_items_seen": int(seen_mask.sum()),
        "recommendations": recs,
    }


def similar_items_collaborative(model: dict, item_id: Any, top_n: int = 10) -> dict:
    """'Users who liked this also liked...' — item-item similarity from interactions."""
    item_ids = model["_item_ids"]
    key = str(item_id)
    if key not in set(item_ids):
        raise ValueError(f"Unknown item '{item_id}'.")
    idx = int(np.where(item_ids == key)[0][0])
    sims = model["_item_similarity"][idx]
    ranked = np.argsort(-sims)
    out = []
    for i in ranked:
        if i == idx:
            continue
        out.append({"item": str(item_ids[i]), "similarity": round(float(sims[i]), 4)})
        if len(out) >= top_n:
            break
    return {"item": key, "similar_items": out}


# ── Content-based filtering ─────────────────────────────────────────────

def fit_content(
    df: pd.DataFrame,
    item_col: str,
    feature_cols: list[str],
    text_cols: Optional[list[str]] = None,
    list_cols: Optional[list[str]] = None,
) -> dict:
    """
    Fit a content-based 'more like this' model over item feature columns.
    - text_cols: free-text columns (description, overview...) -> TF-IDF
    - list_cols: columns holding delimited lists (genres, tags like "Action|Comedy"
      or "Action, Comedy") -> multi-hot encoded
    - everything else in feature_cols: numeric -> scaled, low-cardinality
      categorical -> one-hot
    """
    work = df[[item_col] + feature_cols].dropna(subset=[item_col]).copy()
    work = work.drop_duplicates(subset=[item_col]).reset_index(drop=True)
    item_names = work[item_col].astype(str).to_numpy()

    text_cols = text_cols or []
    list_cols = list_cols or []
    remaining = [c for c in feature_cols if c not in text_cols and c not in list_cols]

    blocks = []

    for c in text_cols:
        texts = _as_plain_str_series(work[c], "")
        if texts.str.strip().eq("").all():
            continue
        tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
        blocks.append(tfidf.fit_transform(texts).toarray())

    for c in list_cols:
        raw = _as_plain_str_series(work[c], "")
        parsed = raw.apply(lambda s: [t.strip() for t in s.replace("|", ",").split(",") if t.strip()])
        mlb = MultiLabelBinarizer()
        blocks.append(mlb.fit_transform(parsed).astype(float))

    numeric_cols = [c for c in remaining if pd.api.types.is_numeric_dtype(work[c])]
    cat_cols = [c for c in remaining if c not in numeric_cols]

    if numeric_cols:
        num = work[numeric_cols].astype(float)
        num = num.fillna(num.median())
        blocks.append(StandardScaler().fit_transform(num))

    for c in cat_cols:
        if work[c].nunique() > 50:
            continue  # skip free-text-like / high-cardinality categoricals left unflagged
        dummies = pd.get_dummies(_as_plain_str_series(work[c], "missing"))
        blocks.append(dummies.to_numpy(dtype=float))

    if not blocks:
        raise ValueError("No usable feature columns found (need text, list/genre, numeric, or low-cardinality categorical columns).")

    feature_matrix = np.hstack(blocks)
    sim = cosine_similarity(feature_matrix)

    return {
        "mode": "content",
        "item_col": item_col,
        "feature_cols": feature_cols,
        "text_cols": text_cols,
        "list_cols": list_cols,
        "n_items": int(len(item_names)),
        "n_features": int(feature_matrix.shape[1]),
        "_item_names": item_names,
        "_similarity": sim,
    }


def recommend_similar(model: dict, item_name: Any, top_n: int = 10) -> dict:
    """'More like this' — content-based recommendations for one item."""
    names = model["_item_names"]
    key = str(item_name)
    matches = np.where(names == key)[0]
    if len(matches) == 0:
        raise ValueError(f"Unknown item '{item_name}'.")
    idx = int(matches[0])
    sims = model["_similarity"][idx]
    ranked = np.argsort(-sims)
    out = []
    for i in ranked:
        if i == idx:
            continue
        out.append({"item": str(names[i]), "similarity": round(float(sims[i]), 4)})
        if len(out) >= top_n:
            break
    return {"item": key, "recommendations": out}


# ── Lightweight evaluation (optional, mirrors automl.py's spirit) ───────

def evaluate_collaborative(df: pd.DataFrame, user_col: str, item_col: str, rating_col: str,
                            test_size: float = 0.2, n_components: int = 20) -> dict:
    """
    Hold out a slice of (user,item,rating) rows and report how well SVD
    reconstructs them — a quick, honest sanity check (RMSE/MAE), same
    spirit as automl.py's train/test reporting.
    """
    work = df[[user_col, item_col, rating_col]].dropna()
    if len(work) < 20:
        raise ValueError("Need at least ~20 ratings to evaluate a collaborative model.")

    shuffled = work.sample(frac=1.0, random_state=42)
    n_test = max(1, int(len(shuffled) * test_size))
    test_df, train_df = shuffled.iloc[:n_test], shuffled.iloc[n_test:]

    model = fit_collaborative(train_df, user_col, item_col, rating_col, n_components=n_components)
    user_ids, item_ids = model["_user_ids"], model["_item_ids"]
    user_pos = {u: i for i, u in enumerate(user_ids)}
    item_pos = {it: i for i, it in enumerate(item_ids)}
    global_mean = float(train_df[rating_col].mean())

    errors = []
    for _, row in test_df.iterrows():
        u, it, actual = str(row[user_col]), str(row[item_col]), float(row[rating_col])
        if u in user_pos and it in item_pos:
            pred = float(model["_user_factors"][user_pos[u]] @ model["_item_factors"][item_pos[it]])
        else:
            pred = global_mean  # cold-start fallback
        errors.append(pred - actual)

    errors = np.array(errors)
    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "rmse": round(float(np.sqrt((errors ** 2).mean())), 4),
        "mae": round(float(np.abs(errors).mean()), 4),
        "baseline_rmse_global_mean": round(
            float(np.sqrt(((test_df[rating_col].astype(float) - global_mean) ** 2).mean())), 4
        ),
        "note": "Lower RMSE/MAE is better. Compare against baseline_rmse_global_mean (predicting the average rating for everyone) to see how much the model actually learned.",
    }
