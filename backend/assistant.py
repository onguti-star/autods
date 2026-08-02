"""
Offline dataset assistant.

The app calls this module with the current pandas DataFrame and a user question.
It does not call an external AI API; instead it uses practical data-science
heuristics so it can explain, inspect, study, and ACT on any uploaded dataset locally.
"""
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def _normalise_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


ASSISTANT_KEYWORDS = {
    "analysis", "analyse", "analyze", "average", "bar", "bottom", "boxplot",
    "categorical", "category", "clean", "cluster", "columns", "compare",
    "correlate", "correlation", "correlations", "data", "dataset", "date",
    "describe", "distribution", "duplicates", "dtype", "extreme", "filter",
    "frequency", "group", "histogram", "maximum", "mean", "median", "minimum",
    "missing", "model", "numeric", "outlier", "outliers", "overview",
    "predict", "prediction", "records", "relationship", "remove", "sample",
    "scatter", "schema", "search", "sort", "standardize", "statistics",
    "study", "summary", "target", "total", "unique", "variance",
    "visualization",
}

CLEAN_KEYWORDS = ASSISTANT_KEYWORDS | {
    "add", "boolean", "column", "convert", "create", "delete", "discard", "drop",
    "erase",
    "duplicates", "emails", "extract", "fill", "first", "float", "greater",
    "integer", "keep", "lowercase", "missing", "multiply", "numbers",
    "outliers", "punctuation", "rename", "replace", "reset", "rows",
    "product", "separate", "split", "stopwords", "strip", "text", "trim",
    "uppercase", "urls", "whitespace",
}


def _keyword_typo_threshold(word: str, keyword: str) -> int:
    if min(len(word), len(keyword)) <= 4:
        return 1
    length = max(len(word), len(keyword))
    if length <= 8:
        return 2
    return 3


def _correct_keyword_typos(text: str, keywords: set[str]) -> tuple[str, list[tuple[str, str]]]:
    """Correct likely misspelled command words while leaving column/value text alone.

    This is intentionally conservative: it only corrects standalone words that are
    close to a known command keyword and ignores short words where false positives
    are common.
    """
    corrections: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def replace(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in keywords or len(lower) < 4 or lower.isdigit():
            return word

        best_keyword = None
        best_distance = None
        for keyword in keywords:
            if abs(len(keyword) - len(lower)) > _keyword_typo_threshold(lower, keyword):
                continue
            distance = _levenshtein(lower, keyword)
            threshold = _keyword_typo_threshold(lower, keyword)
            similarity = 1 - distance / max(len(lower), len(keyword), 1)
            if distance <= threshold and similarity >= 0.65:
                if best_distance is None or distance < best_distance or (
                    distance == best_distance and len(keyword) > len(best_keyword or "")
                ):
                    best_keyword = keyword
                    best_distance = distance

        if not best_keyword:
            return word

        pair = (lower, best_keyword)
        if pair not in seen:
            corrections.append(pair)
            seen.add(pair)
        return best_keyword

    corrected = re.sub(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", replace, text)
    return corrected, corrections


def _format_keyword_correction_note(corrections: list[tuple[str, str]]) -> str:
    if not corrections:
        return ""
    shown = corrections[:3]
    if len(shown) == 1:
        wrong, right = shown[0]
        return f"I think you meant '{right}' instead of '{wrong}'. "
    bits = [f"'{right}' instead of '{wrong}'" for wrong, right in shown]
    return "I corrected likely typos: " + "; ".join(bits) + ". "


def _find_columns_in_text(text: str, columns: list) -> list:
    """Match column names in free text.
    First tries exact normalised match, then falls back to edit-distance
    fuzzy matching (>=0.5 similarity) so spelling mistakes and variations
    like 'unitsold' matching 'units_sold' are handled.
    """
    text_norm = f" {_normalise_text(text)} "
    found = []
    for col in sorted(columns, key=lambda c: len(str(c)), reverse=True):
        col_norm = _normalise_text(col)
        if f" {col_norm} " in text_norm and col not in found:
            found.append(col)

    # Fuzzy fallback — try each unmatched column
    for col in sorted(columns, key=lambda c: len(str(c)), reverse=True):
        if col in found:
            continue
        col_norm = _normalise_text(col)
        # Check each word in the column name against the text
        col_words = col_norm.split()
        if not col_words:
            continue
        # Try matching each word individually (fuzzy)
        for cw in col_words:
            if len(cw) <= 2:
                continue  # skip very short words to avoid false matches
            for tw in text_norm.split():
                if len(tw) <= 2:
                    continue
                max_len = max(len(cw), len(tw), 1)
                ratio = 1 - _levenshtein(cw, tw) / max_len
                if ratio >= 0.5 and col not in found:
                    found.append(col)
                    break
            if col in found:
                break

    return found


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _numeric_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include="number").columns.tolist()


def _numeric_like_cols(df: pd.DataFrame, threshold: float = 0.8) -> list:
    numeric = set(_numeric_cols(df))
    numeric_like = []
    for col in df.columns:
        if col in numeric:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        converted = pd.to_numeric(s, errors="coerce")
        if converted.notna().mean() >= threshold:
            numeric_like.append(col)
    return numeric_like


def _categorical_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _numeric_cols(df)]


def _fmt(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fmt_cell(value, max_len: int = 120) -> str:
    text = _fmt(value)
    return text[: max_len - 3] + "..." if len(text) > max_len else text


def _as_datetime(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(s, errors="coerce")
    return parsed if parsed.notna().mean() >= 0.6 else None


def _corr_strength(corr: float) -> str:
    a = abs(corr)
    if a >= 0.7:
        return "strong"
    if a >= 0.4:
        return "moderate"
    return "weak"


def _corr_sentence(a: str, b: str, corr: float) -> str:
    if pd.isna(corr):
        return f"I could not compute a correlation between '{a}' and '{b}' because there is not enough overlapping numeric data."
    if corr > 0:
        direction = "positive"
        meaning = "higher values of one tend to come with higher values of the other"
    elif corr < 0:
        direction = "negative"
        meaning = "higher values of one tend to come with lower values of the other"
    else:
        direction = "near-zero"
        meaning = "there is no clear linear movement between them"
    return f"'{a}' and '{b}' have a {_corr_strength(corr)} {direction} relationship (correlation {corr:.3f}); {meaning}."


def _correlation_table(df: pd.DataFrame, target: str | None = None) -> dict | None:
    """Build a table of correlations for a target column or all numeric pairs.
    Returns {columns, rows} for table rendering, or None if not applicable."""
    numeric = _numeric_cols(df)
    if len(numeric) < 2:
        return None

    corr = df[numeric].corr(numeric_only=True)
    pairs = []

    if target and target in corr.columns:
        for col in corr.columns:
            if col == target:
                continue
            val = corr.loc[target, col]
            if pd.isna(val):
                continue
            pairs.append((col, val, abs(val)))
    else:
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                val = corr.loc[a, b]
                if pd.isna(val):
                    continue
                pairs.append((f"{a} ↔ {b}", val, abs(val)))

    if not pairs:
        return None

    pairs.sort(key=lambda x: x[2], reverse=True)
    rows = []
    for label, val, _ in pairs[:20]:
        val_float = float(val) if hasattr(val, "item") else val
        rows.append({
            "Correlation": label,
            "Value": round(val_float, 3),
            "Strength": _corr_strength(val),
        })
    return {"columns": ["Correlation", "Value", "Strength"], "rows": rows}


def _dataset_overview(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    numeric = _numeric_cols(df)
    categorical = _categorical_cols(df)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    missing_pct = (missing / (rows * cols) * 100) if rows and cols else 0
    parts = [
        f"The dataset has {rows:,} rows and {cols:,} columns.",
        f"It contains {len(numeric)} numeric column(s) and {len(categorical)} non-numeric/categorical column(s).",
        f"There are {missing:,} missing cells ({missing_pct:.1f}% of all cells) and {duplicates:,} duplicate row(s).",
    ]
    if numeric:
        parts.append("Numeric columns: " + ", ".join(map(str, numeric[:12])) + ("..." if len(numeric) > 12 else "") + ".")
    if categorical:
        parts.append("Categorical/text columns: " + ", ".join(map(str, categorical[:12])) + ("..." if len(categorical) > 12 else "") + ".")
    return " ".join(parts)


def _shape_report(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    return (
        f"The dataset has {rows:,} observation(s), also called rows or records, "
        f"and {cols:,} column(s)."
    )


def _format_column_list(cols: list, limit: int = 20) -> str:
    shown = ", ".join(map(str, cols[:limit]))
    return shown + ("..." if len(cols) > limit else "")


def _column_type_report(df: pd.DataFrame, kind: str) -> str:
    numeric = _numeric_cols(df)
    numeric_like = _numeric_like_cols(df)
    categorical = _categorical_cols(df)

    if kind == "numeric":
        if not numeric and not numeric_like:
            return "I did not find any numeric columns in this dataset."
        parts = []
        if numeric:
            parts.append(f"Numeric columns ({len(numeric)}): {_format_column_list(numeric)}.")
        if numeric_like:
            parts.append(
                f"These columns look numeric but are stored as text ({len(numeric_like)}): "
                f"{_format_column_list(numeric_like)}. Cleaning/converting them could make them usable for numeric analysis."
            )
        return " ".join(parts)

    if kind == "categorical":
        if not categorical:
            return "I did not find any non-numeric/categorical columns in this dataset."
        return f"Non-numeric/categorical columns ({len(categorical)}): {_format_column_list(categorical)}."

    date_cols = []
    for col in df.columns:
        if _as_datetime(df[col]) is not None:
            date_cols.append(col)
    if not date_cols:
        return "I did not find any clear date/time columns in this dataset."
    return f"Date/time-like columns ({len(date_cols)}): {_format_column_list(date_cols)}."


def _is_code_request(question: str) -> bool:
    q = _normalise_text(question)
    code_words = {"code", "codes", "script", "python", "notebook", "snippet", "reproduce"}
    return any(w in q.split() for w in code_words) or "show me how" in q or "you used" in q


def _default_clean_options() -> dict:
    return {
        "drop_duplicates": True,
        "fill_missing_numeric": "median",
        "fill_missing_categorical": "mode",
        "drop_high_missing_cols": True,
        "strip_whitespace": True,
        "drop_constant_cols": True,
        "fix_mixed_types": True,
        "fix_column_names": True,
        "remove_outliers": True,
    }


def _dataset_loader_code(filename: str | None) -> str:
    suffix = str(filename or "").rsplit(".", 1)[-1].lower()
    path = filename or "your_dataset.csv"
    if suffix in {"xlsx", "xls"}:
        return f'DATA_PATH = "{path}"\ndf = pd.read_excel(DATA_PATH)'
    if suffix == "json":
        return f'DATA_PATH = "{path}"\ndf = pd.read_json(DATA_PATH)'
    if suffix in {"tsv", "txt"}:
        return f'DATA_PATH = "{path}"\ndf = pd.read_csv(DATA_PATH, sep="\\t")'
    if suffix == "parquet":
        return f'DATA_PATH = "{path}"\ndf = pd.read_parquet(DATA_PATH)'
    return f'DATA_PATH = "{path}"\ndf = pd.read_csv(DATA_PATH)'


def _cleaning_code(filename: str | None, cleaning_log: list | None, chat_clean_log: list | None,
                   clean_options: dict | None) -> str:
    options = {**_default_clean_options(), **(clean_options or {})}
    log_lines = []
    for entry in cleaning_log or []:
        log_lines.append(f"# - {entry}")
    for entry in chat_clean_log or []:
        command = entry.get("command", "")
        message = entry.get("message", "")
        log_lines.append(f"# - Chat command: {command} -> {message}")
    log_block = "\n".join(log_lines) if log_lines else "# No cleaning actions have been run yet; this is the standard AutoDS cleaning template."

    return f"""Here is the Python cleaning code for this dataset.

```python
import numpy as np
import pandas as pd

{_dataset_loader_code(filename)}

options = {options!r}

# Cleaning actions recorded by AutoDS:
{log_block}

cleaned = df.copy()

if options.get("fix_column_names", True):
    cleaned.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in cleaned.columns
    ]

if options.get("strip_whitespace", True):
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].where(
            cleaned[col].isna(),
            cleaned[col].astype(str).str.strip()
        )

if options.get("fix_mixed_types", True):
    for col in cleaned.select_dtypes(include="object").columns:
        converted = pd.to_numeric(cleaned[col], errors="coerce")
        valid_ratio = converted.notna().sum() / max(cleaned[col].notna().sum(), 1)
        if valid_ratio >= 0.9:
            cleaned[col] = converted

if options.get("drop_constant_cols", True):
    constant_cols = [c for c in cleaned.columns if cleaned[c].nunique(dropna=True) <= 1]
    cleaned = cleaned.drop(columns=constant_cols)

if options.get("drop_high_missing_cols", True):
    high_missing_cols = [c for c in cleaned.columns if cleaned[c].isna().mean() > 0.5]
    cleaned = cleaned.drop(columns=high_missing_cols)

if options.get("drop_duplicates", True):
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

num_strategy = options.get("fill_missing_numeric", "median")
if num_strategy != "none":
    for col in cleaned.select_dtypes(include=[np.number]).columns:
        if cleaned[col].isna().any():
            fill_value = cleaned[col].median() if num_strategy == "median" else cleaned[col].mean()
            if num_strategy == "zero":
                fill_value = 0
            cleaned[col] = cleaned[col].fillna(fill_value)

cat_strategy = options.get("fill_missing_categorical", "mode")
if cat_strategy != "none":
    for col in cleaned.select_dtypes(include="object").columns:
        if cleaned[col].isna().any():
            if cat_strategy == "mode" and not cleaned[col].mode(dropna=True).empty:
                fill_value = cleaned[col].mode(dropna=True).iloc[0]
            else:
                fill_value = "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)

if options.get("remove_outliers", True):
    for col in cleaned.select_dtypes(include=[np.number]).columns:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        if iqr != 0:
            low = q1 - 3 * iqr
            high = q3 + 3 * iqr
            cleaned[col] = cleaned[col].clip(low, high)

cleaned.to_csv("cleaned_data.csv", index=False)
print(cleaned.shape)
```
"""


def _requested_chart_type(question: str) -> str | None:
    q = _normalise_text(question)
    if "scatter" in q:
        return "scatter"
    if "bubble" in q:
        return "bubble"
    if "box plot" in q or "boxplot" in q:
        return "boxplot"
    if "histogram" in q or "hist" in q:
        return "histogram"
    if "line chart" in q or "line plot" in q or re.search(r"\bline\b", q):
        return "line"
    if "pie" in q:
        return "pie"
    if "word cloud" in q or "wordcloud" in q:
        return "wordcloud"
    if "word frequency" in q or "word frequencies" in q:
        return "word_frequency"
    if "bar chart" in q or "bar plot" in q or re.search(r"\bbar\b", q):
        return "bar"
    return None


def _pick_visualization(df: pd.DataFrame, visualization: dict | None,
                        requested_type: str | None = None,
                        mentioned: list | None = None) -> dict:
    numeric = _numeric_cols(df)
    categorical = _categorical_cols(df)
    mentioned = mentioned or []

    if requested_type:
        numeric_mentioned = [c for c in mentioned if c in numeric]
        any_mentioned = [c for c in mentioned if c in df.columns]

        if requested_type in {"scatter", "line"}:
            cols = numeric_mentioned or numeric
            if len(cols) >= 2:
                return {"type": requested_type, "x": cols[0], "y": cols[1]}
        if requested_type == "bubble":
            cols = numeric_mentioned or numeric
            if len(cols) >= 3:
                return {"type": "bubble", "x": cols[0], "y": cols[1], "size": cols[2]}
            if len(cols) >= 2:
                return {"type": "scatter", "x": cols[0], "y": cols[1]}
        if requested_type in {"histogram", "boxplot"}:
            col = (numeric_mentioned or numeric or any_mentioned or list(df.columns))[0]
            return {"type": requested_type, "x": col}
        if requested_type in {"bar", "pie", "word_frequency", "wordcloud"}:
            col = (any_mentioned or categorical or list(df.columns))[0]
            return {"type": requested_type, "x": col}

    if visualization and visualization.get("type") and visualization.get("x"):
        return visualization
    if len(numeric) >= 2:
        return {"type": "scatter", "x": numeric[0], "y": numeric[1]}
    if numeric:
        return {"type": "histogram", "x": numeric[0]}
    if categorical:
        return {"type": "bar", "x": categorical[0]}
    return {"type": "bar", "x": str(df.columns[0]) if len(df.columns) else "column_name"}


def _visualization_code(df: pd.DataFrame, filename: str | None, visualization: dict | None,
                        question: str = "") -> str:
    mentioned = _find_columns_in_text(question, list(df.columns))
    chart = _pick_visualization(df, visualization, _requested_chart_type(question), mentioned)
    chart_type = chart.get("type", "bar")
    x = chart.get("x", "column_name")
    y = chart.get("y")
    group = chart.get("group") or chart.get("size")

    if chart_type == "scatter" and y:
        body = f'''plot_df = df[[{x!r}, {y!r}]].dropna()
plt.scatter(plot_df[{x!r}], plot_df[{y!r}], alpha=0.7)
plt.xlabel({x!r})
plt.ylabel({y!r})
plt.title({f"{y} vs {x}"!r})'''
    elif chart_type == "line" and y:
        body = f'''plot_df = df[[{x!r}, {y!r}]].dropna().sort_values({x!r})
plt.plot(plot_df[{x!r}], plot_df[{y!r}], marker="o")
plt.xlabel({x!r})
plt.ylabel({y!r})
plt.title({f"{y} by {x}"!r})'''
    elif chart_type == "bubble" and y and group:
        body = f'''plot_df = df[[{x!r}, {y!r}, {group!r}]].dropna()
sizes = plot_df[{group!r}]
sizes = 30 + 270 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
plt.scatter(plot_df[{x!r}], plot_df[{y!r}], s=sizes, alpha=0.65)
plt.xlabel({x!r})
plt.ylabel({y!r})
plt.title({f"{y} vs {x}, sized by {group}"!r})'''
    elif chart_type == "histogram":
        body = f'''df[{x!r}].dropna().hist(bins=12)
plt.xlabel({x!r})
plt.ylabel("Count")
plt.title({f"Distribution of {x}"!r})'''
    elif chart_type == "pie":
        body = f'''counts = df[{x!r}].value_counts(dropna=True).head(10)
counts.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title({f"Share of {x}"!r})'''
    elif chart_type == "boxplot":
        if group:
            body = f'''df[[{x!r}, {group!r}]].dropna().boxplot(column={x!r}, by={group!r}, rot=45)
plt.suptitle("")
plt.title({f"Box plot of {x} by {group}"!r})
plt.ylabel({x!r})'''
        else:
            body = f'''df[{x!r}].dropna().plot(kind="box")
plt.ylabel({x!r})
plt.title({f"Box plot of {x}"!r})'''
    elif chart_type in {"word_frequency", "wordcloud"}:
        body = f'''from collections import Counter
import re

words = Counter()
for text in df[{x!r}].dropna().astype(str):
    words.update(re.findall(r"[A-Za-z]{{3,}}", text.lower()))

top_words = pd.Series(dict(words.most_common(25)))
top_words.sort_values().plot(kind="barh")
plt.xlabel("Count")
plt.title({f"Most frequent words in {x}"!r})'''
    else:
        body = f'''counts = df[{x!r}].value_counts(dropna=True).head(15)
counts.sort_values().plot(kind="barh")
plt.xlabel("Count")
plt.title({f"Distribution of {x}"!r})'''

    return f"""Here is Python code to recreate the visualization in Matplotlib.

```python
import pandas as pd
import matplotlib.pyplot as plt

{_dataset_loader_code(filename)}

{body}

plt.tight_layout()
plt.show()
```
"""


def _last_unsupervised_method(unsupervised_results: dict | None, key: str, default: str) -> str:
    result = (unsupervised_results or {}).get(key) or {}
    return str(result.get("selected_method") or result.get("method") or default).lower()


def _clustering_code(df: pd.DataFrame, filename: str | None,
                     unsupervised_results: dict | None, question: str) -> str:
    q = _normalise_text(question)
    last = _last_unsupervised_method(unsupervised_results, "clustering", "kmeans")
    if "agglomerative" in q or "hierarchical" in q:
        method = "hierarchical"
    elif "dbscan" in q:
        method = "dbscan"
    elif "auto" in q or "best" in q:
        method = "auto"
    elif "kmeans" in q or "k means" in q or "k-means" in question.lower():
        method = "kmeans"
    elif "hierarchical" in last:
        method = "hierarchical"
    elif "dbscan" in last:
        method = "dbscan"
    else:
        method = "kmeans"

    numeric = _numeric_cols(df)
    k = 3
    clustering = (unsupervised_results or {}).get("clustering") or {}
    if clustering.get("n_clusters"):
        k = int(clustering["n_clusters"])
    elif len(numeric) >= 2 and len(df) > 3:
        k = min(3, len(df) - 1)

    if method == "hierarchical":
        model_code = f'''model = AgglomerativeClustering(n_clusters={k}, linkage="ward")
labels = model.fit_predict(X_scaled)'''
        import_code = "from sklearn.cluster import AgglomerativeClustering"
        title = "agglomerative/hierarchical clustering"
    elif method == "dbscan":
        model_code = '''model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X_scaled)'''
        import_code = "from sklearn.cluster import DBSCAN"
        title = "DBSCAN clustering"
    else:
        model_code = f'''model = KMeans(n_clusters={k}, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)'''
        import_code = "from sklearn.cluster import KMeans"
        title = "K-Means clustering"

    return f"""Here is Python code for {title} using the same preprocessing style as AutoDS.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
{import_code}

{_dataset_loader_code(filename)}

numeric_cols = df.select_dtypes(include="number").columns.drop("cluster_label", errors="ignore").tolist()
if len(numeric_cols) < 2:
    raise ValueError("Need at least 2 numeric columns for clustering.")

X = df[numeric_cols]
X_imputed = SimpleImputer(strategy="median").fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imputed)

{model_code}

df["cluster_label"] = labels
print(df["cluster_label"].value_counts().sort_index())

if len(set(labels)) > 1 and len(set(labels)) < len(labels):
    print("silhouette:", silhouette_score(X_scaled, labels))
    print("calinski_harabasz:", calinski_harabasz_score(X_scaled, labels))
    print("davies_bouldin:", davies_bouldin_score(X_scaled, labels))

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="tab10", alpha=0.75)
plt.xlabel(numeric_cols[0])
plt.ylabel(numeric_cols[1])
plt.title("Cluster visualization")
plt.tight_layout()
plt.show()
```
"""


def _anomaly_code(filename: str | None) -> str:
    return f"""Here is Python code for anomaly detection.

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

{_dataset_loader_code(filename)}

numeric_cols = df.select_dtypes(include="number").columns.drop("cluster_label", errors="ignore").tolist()
X = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
X_scaled = StandardScaler().fit_transform(X)

model = IsolationForest(contamination=0.1, random_state=42)
labels = model.fit_predict(X_scaled)
scores = model.decision_function(X_scaled)

df["anomaly_label"] = labels
df["anomaly_score"] = scores
print(df["anomaly_label"].value_counts())
```
"""


def _reduction_code(filename: str | None) -> str:
    return f"""Here is Python code for dimensionality reduction with PCA.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

{_dataset_loader_code(filename)}

numeric_cols = df.select_dtypes(include="number").columns.drop("cluster_label", errors="ignore").tolist()
X = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

print("Explained variance:", pca.explained_variance_ratio_)
plt.scatter(coords[:, 0], coords[:, 1], alpha=0.75)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA projection")
plt.tight_layout()
plt.show()
```
"""


def _training_code(filename: str | None) -> str:
    return f"""Here is a reusable Python template for model training.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

{_dataset_loader_code(filename)}

TARGET = "your_target_column"
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
])

is_classification = y.dtype == "object" or y.nunique() <= 20
model = RandomForestClassifier(random_state=42) if is_classification else RandomForestRegressor(random_state=42)
pipe = Pipeline([("preprocess", preprocess), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)

if is_classification:
    print("accuracy:", accuracy_score(y_test, preds))
else:
    print("r2:", r2_score(y_test, preds))
```
"""


def _association_code(filename: str | None) -> str:
    return f"""Here is Python code for association rules with Apriori.

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

{_dataset_loader_code(filename)}

categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
if not categorical_cols:
    raise ValueError("Need at least one categorical or boolean column for association rules.")

encoded = pd.get_dummies(df[categorical_cols], prefix_sep="=", dummy_na=False)
binary_cols = [col for col in encoded.columns if encoded[col].nunique() == 2]
transactions = encoded[binary_cols].astype(bool)

frequent_itemsets = apriori(transactions, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

rules = rules.sort_values("lift", ascending=False)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(20))
```
"""


def answer_code_request(df: pd.DataFrame, question: str, filename: str | None = None,
                        cleaning_log: list | None = None, chat_clean_log: list | None = None,
                        clean_options: dict | None = None,
                        visualization: dict | None = None,
                        unsupervised_results: dict | None = None) -> str | None:
    q = _normalise_text(question)
    if not _is_code_request(question):
        return None

    wants_cleaning = any(term in q for term in ("clean", "cleaning", "preprocess", "preprocessing", "data cleaning"))
    requested_chart = _requested_chart_type(question)
    wants_visual = requested_chart is not None or any(term in q for term in ("visual", "visualization", "visualisation", "chart", "plot", "graph"))
    wants_cluster = any(term in q for term in ("cluster", "clustering", "kmeans", "k means", "hierarchical", "agglomerative", "dbscan"))
    wants_anomaly = any(term in q for term in ("anomaly", "outlier detection", "isolation forest", "lof"))
    wants_reduction = any(term in q for term in ("pca", "dimensionality", "dimension reduction", "reduce"))
    wants_training = any(term in q for term in ("train", "training", "model", "automl", "predict"))
    wants_association = any(term in q for term in ("association", "apriori", "frequent item", "market basket", "rules"))

    if wants_cleaning:
        return _cleaning_code(filename, cleaning_log, chat_clean_log, clean_options)
    if wants_cluster:
        return _clustering_code(df, filename, unsupervised_results, question)
    if wants_anomaly:
        return _anomaly_code(filename)
    if wants_reduction:
        return _reduction_code(filename)
    if wants_association:
        return _association_code(filename)
    if wants_training:
        return _training_code(filename)
    if wants_visual:
        return _visualization_code(df, filename, visualization, question)
    return (
        "Which code do you want? I can generate Python for cleaning, visualization, "
        "specific charts like scatter or histogram, clustering, anomaly detection, "
        "PCA/dimensionality reduction, or model training."
    )


def _dtype_report(df: pd.DataFrame) -> str:
    counts = df.dtypes.astype(str).value_counts()
    parts = [f"{dtype}: {int(count):,}" for dtype, count in counts.items()]
    examples = "; ".join(f"{col}: {dtype}" for col, dtype in df.dtypes.astype(str).head(15).items())
    suffix = "..." if df.shape[1] > 15 else ""
    return f"Data types by count: {'; '.join(parts)}. First columns: {examples}{suffix}."


def _describe_table(df: pd.DataFrame, max_cols: int = 30) -> dict | None:
    if df.empty or df.shape[1] == 0:
        return None
    described = df.iloc[:, :max_cols].describe(include="all").replace({np.nan: None})
    columns = ["Statistic"] + [str(c) for c in described.columns]
    rows = []
    for idx, row in described.iterrows():
        row_dict = {"Statistic": str(idx)}
        for col in described.columns:
            value = row[col]
            if pd.isna(value):
                row_dict[str(col)] = None
            elif isinstance(value, pd.Timestamp):
                row_dict[str(col)] = str(value)
            elif hasattr(value, "item"):
                row_dict[str(col)] = value.item()
            else:
                row_dict[str(col)] = value
        rows.append(row_dict)
    return {"columns": columns, "rows": rows}


def _describe_answer(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    numeric = _numeric_cols(df)
    categorical = _categorical_cols(df)
    shown_cols = min(cols, 30)
    note = f" I am showing the first {shown_cols:,} column(s) in the table below." if cols > shown_cols else ""
    return (
        f"Here is a describe-style statistical summary for {rows:,} row(s) and {cols:,} column(s). "
        f"It includes numeric statistics such as mean, std, quartiles, min and max, plus categorical "
        f"statistics such as unique values, top value and frequency where available. "
        f"Numeric columns: {len(numeric):,}; non-numeric columns: {len(categorical):,}.{note}"
    )


def _extract_lookup_value(question: str) -> str | None:
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
    if quoted:
        return quoted[0].strip()

    cleaned = question.strip()
    patterns = [
        r"\b(?:find|search\s+for|look\s+up|lookup)\s+(?:the\s+)?(?:rows?|records?|details?|data|information)?\b\s*(?:of|for|about|with)?\s*(.+)",
        r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:rows?|records?|details?|data|information)\b\s*(?:of|for|about|with)?\s*(.+)",
        r"\b(?:rows?|records?|details?)\b\s+(?:of|for|about|with)\s+(.+)",
        r"\bwhere\s+is\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
            value = re.sub(r"\b(?:in|from)\s+(?:the\s+)?(?:dataset|data set|data|table)\b.*$", "", value, flags=re.IGNORECASE)
            value = re.sub(r"\b(?:and\s+)?show\s+(?:all\s+)?(?:the\s+)?(?:details?|data|information)\b.*$", "", value, flags=re.IGNORECASE)
            value = value.strip(" ?.,:;-")
            return value or None
    return None


def _row_details(row: pd.Series, max_cols: int = 30) -> str:
    items = []
    for col, value in row.iloc[:max_cols].items():
        items.append(f"{col}: {_fmt_cell(value)}")
    if len(row) > max_cols:
        items.append(f"... {len(row) - max_cols} more column(s)")
    return "; ".join(items)


def _clean_condition_text(question: str) -> str:
    text = question.strip()
    text = re.sub(
        r"^\s*(?:find|search\s+for|look\s+up|lookup|show|get)\s+(?:me\s+)?(?:the\s+)?(?:row|rows|record|records|details?|data|information)?\s*(?:where|with|that\s+has|having)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\s*(?:where|with|that\s+has|having)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:in|from)\s+(?:the\s+)?(?:dataset|data set|data|table)\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:and\s+)?show\s+(?:all\s+)?(?:the\s+)?(?:details?|data|information|row|rows)\b.*$", "", text, flags=re.IGNORECASE)
    return text.strip(" ?.,:;-")


def _parse_row_conditions(question: str, columns: list) -> list:
    text = _clean_condition_text(question)
    if not text:
        return []

    conditions = []
    col_patterns = [(col, _normalise_text(col), re.escape(str(col))) for col in sorted(columns, key=lambda c: len(str(c)), reverse=True)]
    operators = r"(?:=|==|is|equals?|contains?)"
    boundary = r"(?=\s+(?:and|&|,)\s+|$)"

    for col, _, col_raw in col_patterns:
        for pattern in (
            rf"\b{col_raw}\b\s*{operators}\s*['\"]?(.+?)['\"]?{boundary}",
            rf"\b{col_raw}\b\s+['\"]?(.+?)['\"]?{boundary}",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" '\"?.,:;-")
                if value:
                    conditions.append((col, value))
                break

    if conditions:
        seen = set()
        deduped = []
        for col, value in conditions:
            if col not in seen:
                deduped.append((col, value))
                seen.add(col)
        return deduped

    parts = [p.strip() for p in re.split(r"\s+(?:and|&)\s+|,", text, flags=re.IGNORECASE) if p.strip()]
    for part in parts:
        for col, col_norm, _ in col_patterns:
            part_norm = _normalise_text(part)
            if part_norm.startswith(col_norm + " "):
                value = part[len(str(col)):].strip(" =:'\"?.,:;-")
                if value:
                    conditions.append((col, value))
                break
    return conditions


def _condition_mask(s: pd.Series, value: str) -> pd.Series:
    value = value.strip()
    if pd.api.types.is_numeric_dtype(s):
        target = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if not pd.isna(target):
            return s == target
    if pd.api.types.is_bool_dtype(s):
        text = value.lower()
        if text in {"true", "t", "yes", "y", "1", "male", "m"}:
            return s == True
        if text in {"false", "f", "no", "n", "0", "female"}:
            return s == False
    wanted = _normalise_text(value)
    return s.astype("string").map(lambda x: wanted == _normalise_text(x) if not pd.isna(x) else False)


def _lookup_rows_by_conditions(df: pd.DataFrame, conditions: list) -> tuple[str, pd.DataFrame | None]:
    if not conditions:
        return "", None

    mask = pd.Series(True, index=df.index)
    for col, value in conditions:
        mask &= _condition_mask(df[col], value)

    matches = df[mask]
    readable = " and ".join(f"{col} = {value}" for col, value in conditions)
    if matches.empty:
        return f"I could not find any row where {readable}.", None

    shown = matches.head(50)
    parts = [f"Found {len(matches):,} matching row(s) where {readable}."]
    if len(matches) > len(shown):
        parts.append(f"Showing the first {len(shown)} in the table below.")
    return " ".join(parts), shown


def _lookup_rows(df: pd.DataFrame, question: str, mentioned: list) -> tuple[str, pd.DataFrame | None]:
    condition_text, condition_table = _lookup_rows_by_conditions(df, _parse_row_conditions(question, list(df.columns)))
    if condition_text:
        return condition_text, condition_table

    value = _extract_lookup_value(question)
    search_cols = mentioned or list(df.columns)
    if value and mentioned:
        for col in mentioned:
            value = re.sub(
                rf"\b(?:in|from|under|inside|column)\s+(?:the\s+)?{re.escape(str(col))}\b",
                " ",
                value,
                flags=re.IGNORECASE,
            )
            value_norm = _normalise_text(value)
            col_norm = _normalise_text(col)
            value_norm = re.sub(rf"\b{re.escape(col_norm)}\b", " ", value_norm).strip()
            if value_norm:
                value = value_norm
                break

    if not value:
        return "Tell me what value to find, for example: find James Martin.", None

    value_norm = _normalise_text(value)
    tokens = [t for t in value_norm.split() if len(t) > 1]
    if not value_norm or not tokens:
        return "Tell me a more specific value to search for.", None

    matches = []
    for idx, row in df.iterrows():
        cells = [_normalise_text(row[col]) for col in search_cols if col in df.columns]
        row_text = " ".join(cells)
        phrase_in_cell = any(value_norm in cell for cell in cells)
        phrase_in_row = value_norm in row_text
        all_tokens_in_row = all(token in row_text for token in tokens)
        if phrase_in_cell or phrase_in_row or all_tokens_in_row:
            score = (3 if phrase_in_cell else 0) + (2 if phrase_in_row else 0) + (1 if all_tokens_in_row else 0)
            matches.append((score, idx, row))

    if not matches:
        col_note = f" in {', '.join(map(str, search_cols))}" if mentioned else ""
        return f"I could not find any row matching '{value}'{col_note}.", None

    matches.sort(key=lambda item: (-item[0], item[1]))
    shown_matches = matches[:50]
    shown_df = df.loc[[idx for _, idx, _ in shown_matches]]
    parts = [f"Found {len(matches):,} matching row(s) for '{value}'."]
    if len(matches) > len(shown_matches):
        parts.append(f"Showing the first {len(shown_matches)} best match(es) in the table below.")
    return " ".join(parts), shown_df


def _column_summary(df: pd.DataFrame, col: str) -> str:
    s = df[col]
    missing = int(s.isna().sum())
    missing_pct = missing / len(s) * 100 if len(s) else 0
    unique = int(s.nunique(dropna=True))
    if _is_numeric(s):
        clean = s.dropna()
        if clean.empty:
            return f"'{col}' is numeric but has no usable non-missing values."
        skew = clean.skew()
        skew_note = "roughly symmetric" if abs(skew) < 0.5 else "right-skewed" if skew > 0 else "left-skewed"
        return (
            f"'{col}' is numeric with {unique:,} unique values and {missing:,} missing values ({missing_pct:.1f}%). "
            f"Mean {_fmt(clean.mean())}, median {_fmt(clean.median())}, min {_fmt(clean.min())}, max {_fmt(clean.max())}, "
            f"standard deviation {_fmt(clean.std())}. Its distribution is {skew_note} (skew {skew:.2f})."
        )
    top = s.value_counts(dropna=True).head(5)
    if top.empty:
        return f"'{col}' has {unique:,} unique values and {missing:,} missing values, but no non-missing values to summarize."
    top_text = "; ".join(f"{idx}: {int(v):,}" for idx, v in top.items())
    return f"'{col}' is categorical/text with {unique:,} unique values and {missing:,} missing values ({missing_pct:.1f}%). Top values: {top_text}."


def _relationship(df: pd.DataFrame, a: str, b: str) -> str:
    s1, s2 = df[a], df[b]
    if _is_numeric(s1) and _is_numeric(s2):
        return _corr_sentence(a, b, s1.corr(s2))

    dt1, dt2 = _as_datetime(s1), _as_datetime(s2)
    if _is_numeric(s1) and dt2 is not None:
        tmp = pd.DataFrame({a: s1, b: dt2}).dropna().sort_values(b)
        if len(tmp) < 3:
            return f"There is not enough dated data to describe how '{a}' changes over '{b}'."
        k = max(1, len(tmp) // 5)
        first = tmp.iloc[:k][a].mean()
        last = tmp.iloc[-k:][a].mean()
        return f"Over '{b}', '{a}' changes from an early average of {_fmt(first)} to a later average of {_fmt(last)}."
    if _is_numeric(s2) and dt1 is not None:
        return _relationship(df, b, a)

    if _is_numeric(s1) and not _is_numeric(s2):
        grouped = s1.groupby(s2).agg(["mean", "count"]).dropna().sort_values("mean", ascending=False).head(8)
        if grouped.empty:
            return f"I could not compare '{a}' across '{b}' because there are no usable groups."
        parts = [f"{idx}: mean {_fmt(row['mean'])} ({int(row['count']):,} rows)" for idx, row in grouped.iterrows()]
        return f"Average '{a}' by '{b}', highest groups first: " + "; ".join(parts) + "."
    if _is_numeric(s2) and not _is_numeric(s1):
        return _relationship(df, b, a)

    tab = pd.crosstab(s1, s2)
    if tab.empty:
        return f"I could not find enough overlapping values to compare '{a}' and '{b}'."
    pair = tab.stack().sort_values(ascending=False).head(1)
    (top_a, top_b), count = pair.index[0], int(pair.iloc[0])
    return f"The most common '{a}' and '{b}' combination is '{top_a}' with '{top_b}' ({count:,} rows)."


def _top_correlations(df: pd.DataFrame, target: str | None = None) -> str:
    numeric = df[_numeric_cols(df)]
    if numeric.shape[1] < 2:
        return "There are not enough numeric columns to compute correlations."
    corr = numeric.corr(numeric_only=True)
    pairs = []
    if target and target in corr.columns:
        for col in corr.columns:
            if col != target and not pd.isna(corr.loc[target, col]):
                pairs.append((target, col, abs(corr.loc[target, col]), corr.loc[target, col]))
    else:
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                val = corr.loc[a, b]
                if not pd.isna(val):
                    pairs.append((a, b, abs(val), val))
    if not pairs:
        return "I could not compute any usable numeric correlations."
    pairs.sort(key=lambda x: x[2], reverse=True)
    lines = [f"{a} vs {b}: {signed:.3f} ({_corr_strength(signed)})" for a, b, _, signed in pairs[:8]]
    return "Strongest numeric relationships: " + "; ".join(lines) + "."


def _missing_report(df: pd.DataFrame) -> str:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return "No missing values were found in this dataset."
    rows = len(df)
    parts = [f"{col}: {int(n):,} ({n / rows * 100:.1f}%)" for col, n in miss.head(10).items()]
    return "Columns with the most missing values: " + "; ".join(parts) + "."


def _outlier_report(df: pd.DataFrame, col: str | None = None) -> str:
    cols = [col] if col else _numeric_cols(df)
    reports = []
    for c in cols:
        if c not in df.columns or not _is_numeric(df[c]):
            continue
        s = df[c].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < low) | (s > high)).sum())
        if n:
            reports.append((c, n, n / len(s) * 100, low, high))
    if not reports:
        return "I did not find clear IQR-based outliers in the numeric columns."
    reports.sort(key=lambda x: x[1], reverse=True)
    parts = [f"{c}: {n:,} outliers ({pct:.1f}%), outside {_fmt(low)} to {_fmt(high)}" for c, n, pct, low, high in reports[:8]]
    return "Potential outliers: " + "; ".join(parts) + "."


def _distribution_report(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return f"Column '{col}' was not found."
    s = df[col]
    if _is_numeric(s):
        clean = s.dropna()
        if clean.empty:
            return f"'{col}' has no usable numeric values."
        qs = clean.quantile([0.25, 0.5, 0.75])
        return (
            f"Distribution of '{col}': min {_fmt(clean.min())}, Q1 {_fmt(qs.loc[0.25])}, "
            f"median {_fmt(qs.loc[0.5])}, Q3 {_fmt(qs.loc[0.75])}, max {_fmt(clean.max())}, "
            f"skew {clean.skew():.2f}. A histogram or box plot is a good visual for this column."
        )
    top = s.value_counts(dropna=True).head(10)
    parts = [f"{idx}: {int(v):,}" for idx, v in top.items()]
    return f"Distribution of '{col}' by frequency: " + "; ".join(parts) + ". A bar chart is a good visual for this column."


def _group_question(df: pd.DataFrame, mentioned: list) -> str | None:
    if len(mentioned) < 2:
        return None
    numeric = [c for c in mentioned if _is_numeric(df[c])]
    groups = [c for c in mentioned if not _is_numeric(df[c])]
    if numeric and groups:
        return _relationship(df, numeric[0], groups[0])
    return None


def _cluster_feature_cols(df: pd.DataFrame) -> list:
    generated_cols = {"cluster_label", "anomaly_label", "anomaly_score"}
    return [c for c in _numeric_cols(df) if c not in generated_cols]


def _cluster_label_column(df: pd.DataFrame) -> str | None:
    return "cluster_label" if "cluster_label" in df.columns else None


def _cluster_id_from_question(question: str, df: pd.DataFrame) -> int | float | str | None:
    label_col = _cluster_label_column(df)
    if not label_col:
        return None

    q = question.lower()
    match = re.search(r"\bcluster\s*#?\s*(-?\d+(?:\.\d+)?)\b", q)
    if not match:
        return None

    raw = match.group(1)
    wanted = float(raw) if "." in raw else int(raw)
    labels = list(df[label_col].dropna().unique())
    if wanted in labels:
        return wanted

    wanted_text = str(wanted)
    for label in labels:
        if str(label) == wanted_text:
            return label
    return wanted


def _cluster_means(df: pd.DataFrame) -> pd.DataFrame | None:
    label_col = _cluster_label_column(df)
    features = _cluster_feature_cols(df)
    if not label_col or not features:
        return None
    return df.groupby(label_col)[features].mean(numeric_only=True).sort_index()


def _cluster_profile_table(df: pd.DataFrame) -> dict | None:
    means = _cluster_means(df)
    if means is None or means.empty:
        return None
    counts = df.groupby("cluster_label").size().rename("rows")
    profile = means.copy()
    profile.insert(0, "rows", counts)
    profile = profile.reset_index()
    profile.columns = [str(c) for c in profile.columns]
    rows = []
    for _, row in profile.iterrows():
        item = {}
        for col, value in row.items():
            if pd.isna(value):
                item[col] = None
            elif hasattr(value, "item"):
                item[col] = value.item()
            else:
                item[col] = value
        rows.append(item)
    return {"columns": list(profile.columns), "rows": rows}


def _cluster_characteristics(df: pd.DataFrame, question: str) -> str:
    label_col = _cluster_label_column(df)
    if not label_col:
        return "I do not see a 'cluster_label' column yet. Run clustering first, then ask me to describe a cluster."

    features = _cluster_feature_cols(df)
    if not features:
        return "I found cluster labels, but there are no numeric feature columns left to summarize by cluster."

    means = _cluster_means(df)
    if means is None or means.empty:
        return "I could not compute cluster mean values from the current data."

    q = question.lower()
    cluster_id = _cluster_id_from_question(question, df)
    counts = df[label_col].value_counts(dropna=False).sort_index()

    if any(term in q for term in ("largest", "biggest", "most rows", "highest count")) and "mean" not in q:
        label = counts.idxmax()
        return f"Cluster {label} is the largest cluster with {int(counts.loc[label]):,} row(s)."

    if any(term in q for term in ("smallest", "fewest rows", "lowest count")) and "mean" not in q:
        label = counts.idxmin()
        return f"Cluster {label} is the smallest cluster with {int(counts.loc[label]):,} row(s)."

    if any(term in q for term in ("size", "sizes", "how many", "count", "counts", "rows")) and "mean" not in q:
        parts = [f"Cluster {label}: {int(count):,} row(s)" for label, count in counts.items()]
        return "Cluster sizes: " + "; ".join(parts) + "."

    mentioned = [c for c in _find_columns_in_text(question, features) if c in features]
    if mentioned and any(term in q for term in ("highest", "largest", "maximum", "max", "lowest", "smallest", "minimum", "min")):
        col = mentioned[0]
        series = means[col].dropna()
        if series.empty:
            return f"I could not compare clusters by '{col}' because its cluster means are missing."
        if any(term in q for term in ("lowest", "smallest", "minimum", "min")):
            label = series.idxmin()
            direction = "lowest"
        else:
            label = series.idxmax()
            direction = "highest"
        return f"Cluster {label} has the {direction} mean '{col}' at {_fmt(series.loc[label])}."

    if cluster_id is not None:
        if cluster_id not in means.index:
            available = ", ".join(map(str, means.index.tolist()))
            return f"I could not find Cluster {cluster_id}. Available clusters are: {available}."

        cluster_mean = means.loc[cluster_id]
        overall = df[features].mean(numeric_only=True)
        overall_std = df[features].std(numeric_only=True).replace(0, np.nan)
        z = ((cluster_mean - overall) / overall_std).replace([np.inf, -np.inf], np.nan).dropna()
        high = z[z > 0].sort_values(ascending=False).head(3)
        low = z[z < 0].sort_values().head(3)

        rows = int((df[label_col] == cluster_id).sum())
        mean_parts = [f"{col}: {_fmt(cluster_mean[col])}" for col in cluster_mean.index[:12]]
        more = "..." if len(cluster_mean.index) > 12 else ""

        details = [
            f"Cluster {cluster_id} contains {rows:,} row(s).",
            "Mean feature values: " + "; ".join(mean_parts) + more + ".",
        ]
        if not high.empty:
            details.append("Compared with the full dataset, it is highest on: " + "; ".join(f"{col} ({_fmt(cluster_mean[col])})" for col in high.index) + ".")
        if not low.empty:
            details.append("It is lowest on: " + "; ".join(f"{col} ({_fmt(cluster_mean[col])})" for col in low.index) + ".")
        return " ".join(details)

    if any(term in q for term in ("compare", "profile", "profiles", "mean", "means", "characteristics", "describe", "summary")):
        profile = means.copy()
        parts = []
        for label, row in profile.iterrows():
            top = row.dropna().sort_values(ascending=False).head(3)
            summary = "; ".join(f"{col}: {_fmt(value)}" for col, value in top.items())
            parts.append(f"Cluster {label} ({int(counts.get(label, 0)):,} rows) has its highest mean values in {summary}")
        return "Cluster mean profile: " + ". ".join(parts) + ". A table of mean values is shown below."

    return (
        "I can answer cluster questions now. Try: 'describe Cluster 3 based on mean values', "
        "'compare all clusters by feature means', 'which cluster has the highest income?', "
        "or 'show cluster sizes'."
    )


def _is_cluster_question(question: str) -> bool:
    q = _normalise_text(question)
    return any(term in q for term in ("cluster", "clusters", "clustered", "segment", "segments"))


def _recommendations(df: pd.DataFrame) -> str:
    numeric = _numeric_cols(df)
    categorical = _categorical_cols(df)
    ideas = []
    if numeric:
        ideas.append(f"Use histograms/box plots for numeric spread: {', '.join(numeric[:5])}.")
    if len(numeric) >= 2:
        ideas.append(_top_correlations(df))
    if categorical:
        good_cats = [c for c in categorical if 1 < df[c].nunique(dropna=True) <= 20]
        if good_cats:
            ideas.append(f"Use bar charts for category balance: {', '.join(good_cats[:5])}.")
    miss = int(df.isna().sum().sum())
    if miss:
        ideas.append(_missing_report(df))
    outliers = _outlier_report(df)
    if "Potential outliers:" in outliers:
        ideas.append(outliers)
    if not ideas:
        ideas.append("The dataset looks simple; start with column summaries and value counts.")
    return "Study plan: " + " ".join(ideas)


def _target_advice(df: pd.DataFrame, target: str) -> str:
    if target not in df.columns:
        return f"I could not find target column '{target}'."
    s = df[target]
    problem = "regression" if _is_numeric(s) and s.nunique(dropna=True) > 15 else "classification"
    parts = [f"If '{target}' is your target, this looks like a {problem} problem."]
    if problem == "classification":
        counts = s.value_counts(dropna=True).head(8)
        if not counts.empty:
            majority = counts.iloc[0] / counts.sum() * 100
            parts.append(f"Class balance: " + "; ".join(f"{idx}: {int(v):,}" for idx, v in counts.items()) + f". Largest class is {majority:.1f}% of shown rows.")
    else:
        parts.append(_column_summary(df, target))
        parts.append(_top_correlations(df, target))
    feature_candidates = [c for c in df.columns if c != target]
    parts.append(f"Possible feature columns: {', '.join(map(str, feature_candidates[:12]))}" + ("..." if len(feature_candidates) > 12 else "") + ".")
    return " ".join(parts)


def _df_to_table(df: pd.DataFrame, max_rows: int = 50) -> dict:
    """Convert a dataframe slice into a JSON-safe {columns, rows} structure the
    frontend can render as an actual table. Includes the original row index
    as a 'Row #' column, matching the old 'Row {idx}: ...' text format."""
    shown = df.head(max_rows)
    columns = ["Row #"] + [str(c) for c in shown.columns]
    rows = []
    for idx, row in shown.iterrows():
        row_index = idx.item() if hasattr(idx, "item") else idx
        if isinstance(row_index, int):
            row_index += 1   # 1-based, matching how people count rows (not pandas' 0-based index)
        row_dict = {"Row #": row_index}
        for col in shown.columns:
            v = row[col]
            if pd.isna(v):
                row_dict[str(col)] = None
            elif isinstance(v, pd.Timestamp):
                row_dict[str(col)] = str(v)
            elif hasattr(v, "item"):  # numpy scalar (int64, float64, bool_, ...)
                row_dict[str(col)] = v.item()
            else:
                row_dict[str(col)] = v
        rows.append(row_dict)
    return {"columns": columns, "rows": rows}


def answer_question_table(df: pd.DataFrame, question: str) -> dict | None:
    """If the question is a row lookup ('find ...', 'show me the row where ...', etc.)
    and it matched real rows, return them as {columns, rows} for table rendering.
    Returns None for every other kind of question, or when a lookup found nothing."""
    corrected_question, _ = _correct_keyword_typos(question.strip(), ASSISTANT_KEYWORDS)
    q = corrected_question.lower()
    if not q:
        return None
    if _is_cluster_question(question) and any(term in q for term in ("mean", "means", "profile", "profiles", "characteristic", "characteristics", "describe", "compare")):
        return _cluster_profile_table(df)
    if (
        re.search(r"\bdescribe\b", q)
        or "descriptive statistics" in q
        or "summary statistics" in q
        or "statistical summary" in q
    ):
        return _describe_table(df)
    relationship_words = ("relationship", "correlation", "correlate", "associated", "compare", "related")
    if any(w in q for w in relationship_words):
        mentioned = _find_columns_in_text(corrected_question, list(df.columns))
        target = mentioned[0] if mentioned else None
        table = _correlation_table(df, target)
        if table:
            return table
    lookup_question = (
        re.search(r"\b(find|search for|look up|lookup)\b", q)
        or re.search(r"\b(show|get)\s+(me\s+)?(the\s+)?(row|record|details|data|information)\b", q)
        or re.search(r"\b(row|record|details)\s+(of|for|about|with)\b", q)
        or re.search(r"\bwhere\s+is\b", q)
    )
    if not lookup_question:
        return None
    mentioned = _find_columns_in_text(corrected_question, list(df.columns))
    _, matches = _lookup_rows(df, corrected_question, mentioned)
    if matches is None or matches.empty:
        return None
    return _df_to_table(matches)


def execute_action(df: pd.DataFrame, question: str) -> dict:
    """Execute an action on the dataframe based on the user's question.
    Returns a dict with:
      - action: str describing what was done
      - success: bool
      - message: str human-readable result
      - table: dict | None with {columns, rows} if there's data to show
      - modified_df: pd.DataFrame | None if the dataframe was changed
    """
    corrected_question, corrections = _correct_keyword_typos(question.strip(), ASSISTANT_KEYWORDS)
    correction_note = _format_keyword_correction_note(corrections)
    q = corrected_question.lower()
    columns = list(df.columns)
    mentioned = _find_columns_in_text(corrected_question, columns)
    
    result = {
        "action": "none",
        "success": False,
        "message": "",
        "table": None,
        "modified_df": None,
    }

    # --- Remove duplicates ---
    if "remove duplicate" in q or "drop duplicate" in q:
        before = len(df)
        modified = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(modified)
        result["action"] = "remove_duplicates"
        result["success"] = True
        result["message"] = correction_note + f"Removed {removed:,} duplicate row(s). Dataset went from {before:,} to {len(modified):,} rows."
        result["modified_df"] = modified
        return result

    # --- Drop column ---
    if "drop column" in q or "remove column" in q or "delete column" in q:
        if mentioned:
            col = mentioned[0]
            if col in df.columns:
                modified = df.drop(columns=[col])
                result["action"] = "drop_column"
                result["success"] = True
                result["message"] = correction_note + f"Dropped column '{col}'. Dataset now has {len(modified.columns)} columns and {len(modified):,} rows."
                result["modified_df"] = modified
                return result
            else:
                result["message"] = f"Column '{col}' not found."
                return result
        else:
            result["message"] = "Please specify which column to drop. Example: 'drop column age'"
            return result

    # --- Fill missing values ---
    if "fill missing" in q or "fill null" in q or "fill nan" in q:
        if mentioned:
            col = mentioned[0]
            if col not in df.columns:
                result["message"] = f"Column '{col}' not found."
                return result
            s = df[col]
            if _is_numeric(s):
                median_val = s.median()
                modified = df.copy()
                modified[col] = modified[col].fillna(median_val)
                n_filled = int(s.isna().sum())
                result["action"] = "fill_missing"
                result["success"] = True
                result["message"] = correction_note + f"Filled {n_filled:,} missing values in '{col}' with median ({_fmt(median_val)})."
                result["modified_df"] = modified
                return result
            else:
                mode_vals = s.mode(dropna=True)
                fill_val = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
                modified = df.copy()
                modified[col] = modified[col].fillna(fill_val)
                n_filled = int(s.isna().sum())
                result["action"] = "fill_missing"
                result["success"] = True
                result["message"] = correction_note + f"Filled {n_filled:,} missing values in '{col}' with mode ('{fill_val}')."
                result["modified_df"] = modified
                return result
        else:
            result["message"] = "Please specify which column to fill. Example: 'fill missing values in age'"
            return result

    # --- Remove outliers ---
    if "remove outlier" in q or "cap outlier" in q:
        if mentioned and mentioned[0] in df.columns:
            col = mentioned[0]
            if not _is_numeric(df[col]):
                result["message"] = f"Column '{col}' is not numeric, so I cannot detect outliers in it."
                return result
            s = df[col].dropna()
            if len(s) < 4:
                result["message"] = f"Not enough data in '{col}' to detect outliers."
                return result
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                result["message"] = f"All values in '{col}' are the same, so there are no outliers."
                return result
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (df[col] >= low) & (df[col] <= high)
            modified = df[mask].reset_index(drop=True)
            n_removed = len(df) - len(modified)
            result["action"] = "remove_outliers"
            result["success"] = True
            result["message"] = correction_note + f"Capped {n_removed:,} outlier(s) in '{col}' to the range {_fmt(low)} to {_fmt(high)} (1.5×IQR). Dataset now has {len(modified):,} rows."
            result["modified_df"] = modified
            return result
        else:
            # Apply to all numeric columns
            numeric_cols = _numeric_cols(df)
            if not numeric_cols:
                result["message"] = "No numeric columns found to remove outliers from."
                return result
            modified = df.copy()
            total_removed = 0
            for col in numeric_cols:
                s = df[col].dropna()
                if len(s) < 4:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (modified[col] >= low) & (modified[col] <= high)
                total_removed += len(modified) - mask.sum()
                modified = modified[mask]
            modified = modified.reset_index(drop=True)
            result["action"] = "remove_outliers"
            result["success"] = True
            result["message"] = correction_note + f"Capped {total_removed:,} outlier(s) across {len(numeric_cols)} numeric column(s) using 1.5×IQR. Dataset now has {len(modified):,} rows."
            result["modified_df"] = modified
            return result

    # --- Trim whitespace ---
    if "trim whitespace" in q or "strip whitespace" in q or "trim spaces" in q:
        if mentioned:
            cols_to_trim = [c for c in mentioned if c in df.columns and df[c].dtype == 'object']
        else:
            cols_to_trim = df.select_dtypes(include=['object']).columns.tolist()
        
        if not cols_to_trim:
            result["message"] = "No text columns found to trim whitespace from."
            return result
        
        modified = df.copy()
        for col in cols_to_trim:
            modified[col] = modified[col].astype(str).str.strip()
            modified[col] = modified[col].replace('nan', np.nan)
        
        result["action"] = "trim_whitespace"
        result["success"] = True
        result["message"] = correction_note + f"Trimmed leading/trailing whitespace in {len(cols_to_trim)} column(s): {', '.join(cols_to_trim[:5])}."
        result["modified_df"] = modified
        return result

    # --- Standardize column names ---
    if "standardize column" in q or "normalize column" in q or "clean column names" in q:
        old_names = list(df.columns)
        new_names = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in old_names]
        renamed = sum(1 for o, n in zip(old_names, new_names) if o != n)
        if renamed == 0:
            result["message"] = "All column names are already standardized."
            return result
        modified = df.copy()
        modified.columns = new_names
        result["action"] = "standardize_columns"
        result["success"] = True
        result["message"] = correction_note + f"Standardized {renamed:,} column name(s) (lowercase, underscores)."
        result["modified_df"] = modified
        return result

    # --- Sort data ---
    if "sort by" in q or "sort data" in q or "order by" in q:
        if mentioned and mentioned[0] in df.columns:
            col = mentioned[0]
            ascending = "descending" not in q and "desc" not in q
            modified = df.sort_values(by=col, ascending=ascending).reset_index(drop=True)
            direction = "ascending" if ascending else "descending"
            result["action"] = "sort_data"
            result["success"] = True
            result["message"] = correction_note + f"Sorted dataset by '{col}' in {direction} order. Showing first 10 rows:"
            result["table"] = _df_to_table(modified.head(10))
            result["modified_df"] = modified
            return result
        else:
            result["message"] = "Please specify which column to sort by. Example: 'sort by age' or 'sort by price descending'"
            return result

    # --- Filter rows ---
    if "filter" in q or "show only" in q or "keep only" in q or "where" in q:
        # Try to parse conditions
        conditions = _parse_row_conditions(corrected_question, columns)
        if conditions:
            mask = pd.Series(True, index=df.index)
            for col, value in conditions:
                mask &= _condition_mask(df[col], value)
            modified = df[mask].reset_index(drop=True)
            cond_str = " and ".join(f"{c} = {v}" for c, v in conditions)
            result["action"] = "filter_rows"
            result["success"] = True
            result["message"] = correction_note + f"Filtered to {len(modified):,} rows where {cond_str}."
            if len(modified) > 0:
                result["table"] = _df_to_table(modified.head(10))
            result["modified_df"] = modified
            return result
        else:
            result["message"] = "I couldn't parse the filter conditions. Try: 'filter status = active' or 'show only rows where age > 30'"
            return result

    # --- Show top/bottom N ---
    if "top" in q or "bottom" in q or "first" in q or "last" in q:
        if mentioned and mentioned[0] in df.columns:
            col = mentioned[0]
            n = 10  # default
            # Extract number from question
            num_match = re.search(r'\b(\d+)\b', q)
            if num_match:
                n = min(int(num_match.group(1)), 50)
            
            if "bottom" in q or "last" in q:
                subset = df.nsmallest(n, col) if _is_numeric(df[col]) else df.tail(n)
                result["action"] = "show_bottom"
                result["success"] = True
                result["message"] = correction_note + f"Bottom {len(subset)} rows by '{col}':"
            else:
                subset = df.nlargest(n, col) if _is_numeric(df[col]) else df.head(n)
                result["action"] = "show_top"
                result["success"] = True
                result["message"] = correction_note + f"Top {len(subset)} rows by '{col}':"
            
            result["table"] = _df_to_table(subset)
            return result

    # --- Random sample ---
    if "random sample" in q or "sample" in q:
        n_match = re.search(r'\b(\d+)\b', q)
        n = int(n_match.group(1)) if n_match else 10
        n = min(n, 50)
        modified = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
        result["action"] = "random_sample"
        result["success"] = True
        result["message"] = correction_note + f"Random sample of {len(modified)} rows:"
        result["table"] = _df_to_table(modified)
        result["modified_df"] = modified
        return result

    # --- Reset to original ---
    if "reset" in q or "undo all" in q or "revert all" in q:
        result["action"] = "reset"
        result["success"] = True
        result["message"] = "Use the Undo button in the cleaning panel to revert changes step by step, or re-upload the file to start fresh."
        return result

    # No action recognized
    result["message"] = None  # Signal that this is not an action question
    return result



def _data_suggestions(df: pd.DataFrame) -> str:
    """
    Rich, specific, actionable suggestions tailored to what's actually in the data.
    Triggered by phrases like 'give me suggestions', 'what can I do with this data', etc.
    """
    numeric   = _numeric_cols(df)
    categoric = _categorical_cols(df)
    rows, cols = df.shape
    missing_total = int(df.isna().sum().sum())
    missing_cols  = [c for c in df.columns if df[c].isna().any()]
    dup_rows      = int(df.duplicated().sum())

    parts = []

    # --- Header ---
    parts.append(f"Here's what I'd suggest for your dataset ({rows:,} rows, {cols} columns):\n")

    # --- Data Quality ---
    quality_items = []
    if missing_total:
        pct = round(100 * missing_total / (rows * cols), 1)
        cols_list = ", ".join(f"'{c}'" for c in missing_cols[:6]) + ("..." if len(missing_cols) > 6 else "")
        quality_items.append(f"Fill or remove missing values — {missing_total:,} missing cells ({pct}%) across: {cols_list}.")
    if dup_rows:
        quality_items.append(f"Remove {dup_rows:,} duplicate rows.")
    # Check for object columns that might be numeric
    for c in categoric:
        try:
            pd.to_numeric(df[c].dropna().head(50), errors="raise")
            quality_items.append(f"Convert '{c}' to a number — it looks numeric but is stored as text.")
            break
        except Exception:
            pass
    if quality_items:
        parts.append("🧹 Data Cleaning:")
        parts.extend(f"  • {item}" for item in quality_items)
        parts.append("")

    # --- Visualisation suggestions ---
    viz_items = []
    if numeric:
        viz_items.append(f"Plot distributions (histogram or box plot) for: {', '.join(numeric[:5])}{'...' if len(numeric) > 5 else ''}.")
    if len(numeric) >= 2:
        try:
            corr = df[numeric].corr().abs()
            pairs = []
            for i in range(len(numeric)):
                for j in range(i + 1, len(numeric)):
                    pairs.append((corr.iloc[i, j], numeric[i], numeric[j]))
            pairs.sort(reverse=True)
            if pairs and pairs[0][0] > 0.3:
                top = pairs[0]
                viz_items.append(f"Scatter plot '{top[1]}' vs '{top[2]}' — they have a {top[0]:.0%} correlation.")
        except Exception:
            pass
    good_cats = [c for c in categoric if 1 < df[c].nunique(dropna=True) <= 20]
    if good_cats:
        viz_items.append(f"Bar chart to see category balance in: {', '.join(good_cats[:4])}{'...' if len(good_cats) > 4 else ''}.")
    if viz_items:
        parts.append("📊 Visualisation:")
        parts.extend(f"  • {item}" for item in viz_items)
        parts.append("")

    # --- Modelling suggestions ---
    model_items = []
    # Good classification targets: categorical with 2-10 unique values
    clf_targets = [c for c in categoric if 2 <= df[c].nunique(dropna=True) <= 10]
    # Good regression targets: numeric with high variance
    reg_targets = [c for c in numeric if df[c].std() > 0 and df[c].nunique(dropna=True) > 15]

    if clf_targets:
        model_items.append(
            f"Classification — try predicting '{clf_targets[0]}' "
            f"({'or ' + chr(39) + clf_targets[1] + chr(39) if len(clf_targets) > 1 else ''}) "
            f"using the other columns as features."
        )
    if reg_targets:
        model_items.append(
            f"Regression — try predicting '{reg_targets[0]}' as a numeric value."
        )
    if len(numeric) >= 3 and rows >= 50:
        model_items.append(
            "Clustering — run unsupervised learning to find natural groups in the data."
        )
    if not model_items:
        model_items.append("Add more rows or numeric columns to unlock supervised modelling.")

    parts.append("🤖 Modelling:")
    parts.extend(f"  • {item}" for item in model_items)
    parts.append("")

    # --- Quick wins ---
    quick = []
    if missing_cols:
        quick.append(f'In the clean assistant, type: fill missing in {missing_cols[0]} with median')
    if good_cats and numeric:
        quick.append(f'In the assistant, ask: show average {numeric[0]} by {good_cats[0]}')
    if clf_targets or reg_targets:
        target = (clf_targets or reg_targets)[0]
        quick.append(f"On the Train tab, choose '{target}' as your target column and hit Train.")
    if quick:
        parts.append("⚡ Quick wins to try now:")
        parts.extend(f"  • {q}" for q in quick)

    return "\n".join(parts)

def answer_question(df: pd.DataFrame, question: str) -> str:
    corrected_question, corrections = _correct_keyword_typos(question.strip(), ASSISTANT_KEYWORDS)
    note = _format_keyword_correction_note(corrections)
    return note + _answer_question_impl(df, corrected_question)


def _answer_question_impl(df: pd.DataFrame, question: str) -> str:
    q = question.strip().lower()
    columns = list(df.columns)
    mentioned = _find_columns_in_text(question, columns)

    if not q:
        return "Ask me about this dataset: summary, missing values, correlations, outliers, distributions, grouped averages, or a possible prediction target."

    if _is_cluster_question(question):
        return _cluster_characteristics(df, question)

    row_count_question = (
        re.search(r"\b(how many|number of|total number of|count of|total)\s+(rows?|records?|observations?|entries|samples?)\b", q)
        or re.search(r"\b(rows?|records?|observations?|entries|samples?)\s+(are there|in (the )?(dataset|data set|data))\b", q)
        or q in {"rows", "records", "observations", "shape", "size"}
    )
    if row_count_question:
        return _shape_report(df)

    if (
        re.search(r"\bdescribe\b", q)
        or "descriptive statistics" in q
        or "summary statistics" in q
        or "statistical summary" in q
    ):
        return _describe_answer(df)

    if any(w in q for w in ("overview", "summarize dataset", "summary of dataset", "study", "understand", "explain dataset", "what is in this data")):
        return _dataset_overview(df) + " " + _recommendations(df)

    type_question = any(
        phrase in q
        for phrase in (
            "what column",
            "what columns",
            "which column",
            "which columns",
            "list column",
            "list columns",
            "show column",
            "show columns",
            "column contains",
            "columns contain",
            "column has",
            "columns have",
        )
    )
    if type_question:
        if any(w in q for w in ("numeric", "number", "numbers", "integer", "float", "decimal", "continuous")):
            return _column_type_report(df, "numeric")
        if any(w in q for w in ("categorical", "category", "text", "string", "object", "non numeric", "non-numeric")):
            return _column_type_report(df, "categorical")
        if any(w in q for w in ("date", "time", "datetime", "timestamp")):
            return _column_type_report(df, "date")

    if "data type" in q or "dtype" in q or "dtypes" in q or "schema" in q:
        return _dtype_report(df)

    lookup_question = (
        re.search(r"\b(find|search for|look up|lookup)\b", q)
        or re.search(r"\b(show|get)\s+(me\s+)?(the\s+)?(row|record|details|data|information)\b", q)
        or re.search(r"\b(row|record|details)\s+(of|for|about|with)\b", q)
        or re.search(r"\bwhere\s+is\b", q)
    )
    if lookup_question:
        text, _ = _lookup_rows(df, question, mentioned)
        return text

    _suggestion_triggers = (
        "suggest", "suggestion", "suggestions",
        "what can i do", "what should i do", "what can i do with", "what to do",
        "what can be done", "what should be done",
        "give me ideas", "give me suggestions", "give suggestions",
        "how should i", "how do i start", "where do i start", "where should i start",
        "what next", "next steps", "next step", "what is next",
        "help me start", "help me begin", "help me analyse", "help me analyze",
        "what analysis", "what analyses", "what insights",
        "what can you tell me about this data", "tell me about this data",
        "recommend", "what should i", "insight", "eda", "analyse", "analyze",
        "study plan", "data science",
    )
    if any(w in q for w in _suggestion_triggers):
        return _data_suggestions(df)

    if "target" in q or "predict" in q or "prediction" in q or "model" in q:
        if mentioned:
            return _target_advice(df, mentioned[0])
        return "Tell me the target column you want to predict. I can then suggest whether it is classification or regression and which features may matter."

    if "missing" in q or "null" in q or "nan" in q:
        if mentioned:
            col = mentioned[0]
            n = int(df[col].isna().sum())
            pct = n / len(df) * 100 if len(df) else 0
            return f"'{col}' has {n:,} missing value(s), which is {pct:.1f}% of rows."
        return _missing_report(df)

    if "duplicate" in q:
        n = int(df.duplicated().sum())
        return f"The dataset has {n:,} duplicate row(s)."

    if "outlier" in q or "extreme" in q:
        return _outlier_report(df, mentioned[0] if mentioned else None)

    if "distribution" in q or "histogram" in q or "spread" in q or "skew" in q:
        if mentioned:
            return _distribution_report(df, mentioned[0])
        nums = _numeric_cols(df)
        return "Tell me which column to inspect. Good numeric choices: " + ", ".join(map(str, nums[:10])) + "."

    relationship_words = ("relationship", "correlation", "correlate", "associated", "compare", "related")
    is_correlation_question = any(w in q for w in relationship_words)
    
    # Check for "X vs Y" or "X versus Y" format
    vs_pattern = re.search(r"(.+?)\s+(?:vs|versus)\s+(.+)", q)
    if vs_pattern and len(mentioned) >= 2:
        # This is asking for comparison between two specific columns
        return _relationship(df, mentioned[0], mentioned[1])
    
    if is_correlation_question:
        # Check if this is a "how does X correlate" type question (correlation analysis)
        correlation_analysis_pattern = re.search(r"how\s+does\s+(.+?)\s+correlate", q)
        if correlation_analysis_pattern and len(mentioned) >= 1:
            # This is asking for correlation analysis of one column with all others
            return _top_correlations(df, mentioned[0])
        if len(mentioned) >= 2:
            return _relationship(df, mentioned[0], mentioned[1])
        return _top_correlations(df)
    
    # Only show group averages if this wasn't a correlation question
    group_ans = _group_question(df, mentioned)
    explicit_group_request = any(w in q for w in ("by", "group", "per", "against", "across"))
    if group_ans and explicit_group_request:
        return group_ans

    if any(w in q for w in ("average", "avarage", "avg", "mean")):
        if mentioned and _is_numeric(df[mentioned[0]]):
            col = mentioned[0]
            return f"The average '{col}' is {_fmt(df[col].mean())}."
        if group_ans and explicit_group_request:
            return group_ans
        return "Tell me which numeric column you want the average of."

    if "total" in q or "sum" in q:
        if mentioned and _is_numeric(df[mentioned[0]]):
            col = mentioned[0]
            return f"The total of '{col}' is {_fmt(df[col].sum())}."
        return "Tell me which numeric column you want summed."

    if "median" in q:
        if mentioned and _is_numeric(df[mentioned[0]]):
            col = mentioned[0]
            return f"The median '{col}' is {_fmt(df[col].median())}."
        return "Tell me which numeric column you want the median of."

    if "standard deviation" in q or "std" in q or "variance" in q:
        if mentioned and _is_numeric(df[mentioned[0]]):
            col = mentioned[0]
            if "variance" in q:
                return f"The variance of '{col}' is {_fmt(df[col].var())}."
            return f"The standard deviation of '{col}' is {_fmt(df[col].std())}."

    if "maximum" in q or "highest" in q or "top" in q or re.search(r"\bmax\b", q):
        if mentioned and _is_numeric(df[mentioned[0]]):
            col = mentioned[0]
            return f"The maximum '{col}' is {_fmt(df[col].max())}."
    if "minimum" in q or "lowest" in q or "bottom" in q or re.search(r"\bmin\b", q):
        if mentioned and _is_numeric(df[mentioned[0]]):
            col = mentioned[0]
            return f"The minimum '{col}' is {_fmt(df[col].min())}."

    if "unique" in q or "distinct" in q or "how many different" in q or "cardinality" in q:
        if mentioned:
            col = mentioned[0]
            return f"'{col}' has {df[col].nunique(dropna=True):,} unique value(s)."
        cards = df.nunique(dropna=True).sort_values(ascending=False).head(10)
        return "Highest-cardinality columns: " + "; ".join(f"{c}: {int(v):,}" for c, v in cards.items()) + "."

    if "most common" in q or "most frequent" in q or "mode" in q or "value counts" in q or "frequency" in q:
        if mentioned:
            return _distribution_report(df, mentioned[0])
        return "Tell me which column you want value counts for."

    if "how many rows" in q or "how many records" in q or "shape" in q or "size" in q:
        return _shape_report(df)

    if "how many numeric" in q or "number of numeric" in q:
        numeric = _numeric_cols(df)
        numeric_like = _numeric_like_cols(df)
        extra = f" I also found {len(numeric_like):,} numeric-looking text column(s)." if numeric_like else ""
        return f"The dataset has {len(numeric):,} true numeric column(s).{extra}"

    if "how many categorical" in q or "number of categorical" in q or "how many text" in q:
        categorical = _categorical_cols(df)
        return f"The dataset has {len(categorical):,} non-numeric/categorical column(s)."

    if "how many columns" in q or "columns" == q.strip():
        return f"The dataset has {df.shape[1]:,} columns: {', '.join(map(str, columns))}."

    if mentioned:
        return _column_summary(df, mentioned[0])

    return (
        "I can help study this dataset with EDA questions: overview, missing values, duplicates, "
        "correlations, relationships between columns, means/medians/totals, distributions, outliers, "
        "grouped averages, value counts, cluster profiles, target advice, and visualization ideas. Try: "
        "\"study this dataset\", \"strongest correlations\", \"outliers in price\", "
        "\"average sales by region\", \"describe Cluster 3\", or \"can I predict survival?\""
    )
