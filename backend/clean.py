"""
Chat-based dataset cleaning.

Like assistant.py, this does not call an external AI API — it parses plain
English cleaning instructions with regex/heuristics and applies the matching
pandas operation locally. If a command isn't understood, it explains what
phrasings are supported instead of guessing.
"""
import re

import numpy as np
import pandas as pd

from . import nlp
from .assistant import _find_columns_in_text, _normalise_text


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


_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "hundred": 100, "thousand": 1000,
}

def _coerce_numeric(raw: str):
    """Convert a string to a number, including English words like 'four'."""
    raw = raw.strip().lower()
    if raw in _WORD_TO_NUM:
        return _WORD_TO_NUM[raw]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return None


def _find_columns_multi(cols_text: str, columns: list) -> list:
    """
    Match column names from a comma/and-separated string.
    First tries exact normalised match, then falls back to edit-distance
    fuzzy matching (>=0.6 similarity) for each unmatched token.
    This handles typos and slight name mismatches like
    'discount_applied' matching 'discount_allowed'.
    """
    exact = _find_columns_in_text(cols_text, columns)
    found_set = set(exact)

    # Split the typed text into individual column tokens
    import re as _re
    raw_tokens = _re.split(r",|\band\b|\bor\b", cols_text, flags=_re.IGNORECASE)
    raw_tokens = [t.strip() for t in raw_tokens if t.strip()]

    for token in raw_tokens:
        token_norm = _normalise_text(token)
        if not token_norm:
            continue
        # Already matched exactly?
        if any(_normalise_text(c) == token_norm for c in exact):
            continue
        # Find closest column by edit distance
        best_col, best_ratio = None, 0.0
        for col in columns:
            if col in found_set:
                continue
            col_norm = _normalise_text(col)
            max_len = max(len(token_norm), len(col_norm), 1)
            ratio = 1 - _levenshtein(token_norm, col_norm) / max_len
            if ratio > best_ratio:
                best_ratio, best_col = ratio, col
        if best_col and best_ratio >= 0.6:
            found_set.add(best_col)
            exact.append(best_col)

    return exact


_TYPE_ALIASES = {
    "datetime": {"date/time", "datetime", "date", "time"},
    "integer": {"integer", "int", "wholenumber"},
    "float": {"float", "decimal", "float/decimal", "number", "numeric"},
    "text": {"text", "string", "str"},
    "category": {"category", "categorical"},
    "boolean": {"boolean", "bool", "true/false", "yes/no"},
}


def _normalize_type_phrase(raw: str) -> str | None:
    """Match a free-form type phrase (e.g. 'Date / time', 'decimal', 'int') to one of
    integer/float/text/category/datetime/boolean — the same set the dropdown uses."""
    key = re.sub(r"\s+", "", raw.lower().strip(" '\"?.,:;-"))
    for canonical, aliases in _TYPE_ALIASES.items():
        if key in aliases:
            return canonical
    return None


def _convert_type(df: pd.DataFrame, column: str, dtype: str) -> tuple[pd.DataFrame, str]:
    """Convert one column's dtype. Mirrors the structured 'Change column data type' logic
    so chat and the dropdown behave identically. Never raises — returns (df, message);
    on failure the returned df is unchanged."""
    out = df.copy()
    before = str(out[column].dtype)
    before_missing = int(out[column].isna().sum())

    try:
        if dtype == "integer":
            converted = pd.to_numeric(out[column], errors="coerce")
            if bool((converted.dropna() % 1 != 0).any()):
                return df, f"'{column}' contains decimal values, so it can't be safely converted to integer."
            out[column] = converted.astype("Int64")
        elif dtype == "float":
            out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
        elif dtype == "text":
            out[column] = out[column].astype("string")
        elif dtype == "category":
            out[column] = out[column].astype("category")
        elif dtype == "datetime":
            out[column] = pd.to_datetime(out[column], errors="coerce")
        elif dtype == "boolean":
            truthy = {"true", "t", "yes", "y", "1"}
            falsy = {"false", "f", "no", "n", "0"}

            def to_bool(value):
                if pd.isna(value):
                    return pd.NA
                if isinstance(value, bool):
                    return value
                text = str(value).strip().lower()
                if text in truthy:
                    return True
                if text in falsy:
                    return False
                return pd.NA

            out[column] = out[column].map(to_bool).astype("boolean")
        else:
            return df, f"Unsupported target type '{dtype}'."
    except Exception as e:
        return df, f"Could not convert '{column}' to {dtype}: {e}"

    after = str(out[column].dtype)
    after_missing = int(out[column].isna().sum())
    introduced = after_missing - before_missing
    note = f"Changed column '{column}' from {before} to {after}."
    if introduced > 0:
        note += f" {introduced:,} value(s) could not be converted and became missing."
    return out, note


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def _match_col(text: str, columns: list) -> str | None:
    found = _find_columns_in_text(text, columns)
    return found[0] if found else None


def _match_two_cols(text: str, columns: list) -> tuple[str | None, str | None]:
    found = _find_columns_in_text(text, columns)
    if len(found) >= 2:
        return found[0], found[1]
    return (found[0], None) if found else (None, None)


def _parse_value_token(raw: str):
    raw = raw.strip(" '\"?.,:;-")
    try:
        if re.fullmatch(r"-?\d+", raw):
            return int(raw)
        if re.fullmatch(r"-?\d*\.\d+", raw):
            return float(raw)
    except ValueError:
        pass
    return raw


HELP_TEXT = (
    "I understand plain-English cleaning commands. Try things like:\n"
    "• show null rows in order_date  (lists row numbers where a column is missing)\n"
    "• show where rows is null in price\n"
    "• replace 2 with 0 in profit  (replaces exact values in a column)\n"
    "• replace yes with 1 in discount_applied\n"
    "• remove duplicates\n"
    "• remove duplicates of column age\n"
    "• drop column notes\n"
    "• rename column dob to date_of_birth\n"
    "• fill missing values in income with median  (mean / mode / zero / a specific value)\n"
    "• remove rows where email is missing\n"
    "• strip whitespace in name\n"
    "• remove stopwords in review  (the, is, and, ...)\n"
    "• remove punctuation in review\n"
    "• remove urls in comments\n"
    "• remove emails in comments\n"
    "• remove numbers in description\n"
    "• remove Mr. in name  (removes exact text from a column)\n"
    "• lowercase email\n"
    "• uppercase country_code\n"
    "• remove outliers in salary\n"
    "• remove rows where status is cancelled\n"
    "• keep only rows where country is Kenya\n"
    "• convert price to number  (or: integer / text / category / date / time / boolean)\n"
    "• change ORDER_DATE to date / time\n"
    "• reset  (undoes every chat cleaning command and restores the original upload)"
)


def run_command(df: pd.DataFrame, text: str, original_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """
    Parse one natural-language cleaning command and apply it.
    Returns (possibly-new dataframe, human-readable result message).
    The input df is never mutated in place — a modified copy is returned.
    `original_df`, if given, is what "reset" restores.
    """
    q = text.strip()
    ql = q.lower()
    columns = list(df.columns)

    if not q:
        return df, HELP_TEXT

    if ql in ("help", "?", "what can you do", "commands", "what commands"):
        return df, HELP_TEXT

    # ---- reset ----
    if re.search(r"\b(reset|undo everything|restore original|start over|undo all)\b", ql):
        if original_df is None:
            return df, "No original data is available to reset to."
        return original_df.copy(), "Restored the original uploaded data — every chat cleaning command has been undone."

    # ---- remove duplicates ----
    dup_col_match = re.search(r"duplicate\w*\s+(?:of|in|for|values?\s+in)\s+(.+)", ql)
    if "duplicate" in ql:
        col = _match_col(q, columns)
        if col and (dup_col_match or "column" in ql or "of " in ql or "in " in ql):
            before = len(df)
            new_df = df.drop_duplicates(subset=[col]).reset_index(drop=True)
            removed = before - len(new_df)
            if removed == 0:
                return df, f"No duplicate values found in '{col}' — nothing was removed."
            return new_df, f"Removed {removed:,} row(s) that had a duplicate value in '{col}'."
        before = len(df)
        new_df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(new_df)
        if removed == 0:
            return df, "No exact duplicate rows were found — nothing was removed."
        return new_df, f"Removed {removed:,} exact duplicate row(s)."

    # ---- drop column ----
    m = re.search(r"\b(?:drop|remove|delete)\s+(?:the\s+)?column\s+(.+)", ql) or \
        re.search(r"\b(?:drop|remove|delete)\s+(.+?)\s+column\b", ql)
    if m:
        col = _match_col(m.group(1), columns) or _match_col(q, columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        return df.drop(columns=[col]), f"Dropped column '{col}'."

    # ---- rename column ----
    m = re.search(r"\brename\s+(?:column\s+)?(.+?)\s+to\s+(.+)", ql)
    if m:
        old_col = _match_col(m.group(1), columns)
        new_name = m.group(2).strip(" '\"?.,:;-")
        if not old_col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        if not new_name:
            return df, "Tell me the new name, for example: rename dob to date_of_birth."
        return df.rename(columns={old_col: new_name}), f"Renamed column '{old_col}' to '{new_name}'."

    # ---- fill missing (one or many columns) ----
    m = re.search(r"\bfill\s+(?:missing|na|null)\s*(?:values?)?\s*(?:in\s+)?(.+?)\s+with\s+(.+)", ql)
    if m:
        cols_text = m.group(1)
        strategy_raw = m.group(2).strip(" '\"?.,:;-")
        matched_cols = _find_columns_multi(cols_text, columns)
        if not matched_cols:
            return df, f"I couldn't find any columns matching '{cols_text.strip()}'. Available columns: {', '.join(columns)}."
        new_df = df.copy()
        results = []
        skipped = []
        for col in matched_cols:
            n_missing = int(new_df[col].isna().sum())
            if n_missing == 0:
                skipped.append(col)
                continue
            if strategy_raw in ("mean", "average") and pd.api.types.is_numeric_dtype(new_df[col]):
                val = new_df[col].mean()
            elif strategy_raw == "median" and pd.api.types.is_numeric_dtype(new_df[col]):
                val = new_df[col].median()
            elif strategy_raw == "mode":
                mode = new_df[col].mode(dropna=True)
                val = mode.iloc[0] if len(mode) else "Unknown"
            elif strategy_raw in ("zero", "0"):
                val = 0
            else:
                val = _parse_value_token(strategy_raw)
            new_df[col] = new_df[col].fillna(val)
            results.append(f"'{col}': filled {n_missing:,} missing value(s) with {_fmt(val)}")
        if not results:
            return df, f"None of {', '.join(repr(c) for c in matched_cols)} had missing values — nothing to fill."
        msg = "Filled missing values:\n" + "\n".join(f"  • {r}" for r in results)
        if skipped:
            msg += f"\n  (skipped {', '.join(repr(c) for c in skipped)} — already complete)"
        return new_df, msg

    # ---- drop rows where col is missing ----
    m = re.search(r"\b(?:remove|drop)\s+rows?\s+where\s+(.+?)\s+is\s+(?:missing|null|na|empty)\b", ql)
    if m:
        col = _match_col(m.group(1), columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        before = len(df)
        new_df = df.dropna(subset=[col]).reset_index(drop=True)
        removed = before - len(new_df)
        if removed == 0:
            return df, f"'{col}' has no missing values — no rows removed."
        return new_df, f"Removed {removed:,} row(s) where '{col}' was missing."

    # ---- strip whitespace ----
    m = re.search(r"\b(?:strip|trim)\s+(?:whitespace\s+)?(?:in\s+|from\s+)?(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        if col:
            new_df = df.copy()
            new_df[col] = new_df[col].where(new_df[col].isna(), new_df[col].astype(str).str.strip())
            return new_df, f"Trimmed leading/trailing whitespace in '{col}'."

    # ---- lowercase / uppercase ----
    m = re.search(r"\b(lowercase|uppercase)\s+(?:column\s+)?(.+)", ql) or \
        re.search(r"\bmake\s+(.+?)\s+(lowercase|uppercase)\b", ql)
    if m:
        groups = m.groups()
        case_word = groups[0] if groups[0] in ("lowercase", "uppercase") else groups[1]
        col_text = groups[1] if groups[0] in ("lowercase", "uppercase") else groups[0]
        col = _match_col(col_text, columns)
        if col:
            new_df = df.copy()
            if case_word == "lowercase":
                new_df[col] = new_df[col].where(new_df[col].isna(), new_df[col].astype(str).str.lower())
            else:
                new_df[col] = new_df[col].where(new_df[col].isna(), new_df[col].astype(str).str.upper())
            return new_df, f"Converted '{col}' to {case_word}."

    # ---- text cleaning: stopwords / punctuation / urls / emails / numbers ----
    m = re.search(r"\bremove\s+stop\s*words?\s*(?:in\s+|from\s+)?(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        new_df = df.copy()
        new_df[col] = new_df[col].map(nlp.remove_stopwords)
        return new_df, f"Removed common stopwords (the, is, and, ...) from '{col}'."

    m = re.search(r"\bremove\s+(?:punctuation|punct)\s*(?:in\s+|from\s+)?(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        new_df = df.copy()
        new_df[col] = new_df[col].map(nlp.remove_punctuation)
        return new_df, f"Removed punctuation from '{col}'."

    m = re.search(r"\bremove\s+urls?\s*(?:in\s+|from\s+)?(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        new_df = df.copy()
        new_df[col] = new_df[col].map(nlp.remove_urls)
        return new_df, f"Removed URLs from '{col}'."

    m = re.search(r"\bremove\s+emails?\s*(?:in\s+|from\s+)?(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        new_df = df.copy()
        new_df[col] = new_df[col].map(nlp.remove_emails)
        return new_df, f"Removed email addresses from '{col}'."

    m = re.search(r"\bremove\s+numbers?\s*(?:in\s+|from\s+)?(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        new_df = df.copy()
        new_df[col] = new_df[col].map(nlp.remove_numbers)
        return new_df, f"Removed numbers from '{col}'."

    # ---- remove exact text from a column ----
    m = re.search(r"\bremove\s+(.+?)\s+(?:in|from)\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        text_to_remove = m.group(1).strip(" '\"")
        col = _match_col(m.group(2), columns)
        if col and text_to_remove:
            pattern = re.escape(text_to_remove)
            if text_to_remove.endswith(".") and len(text_to_remove) > 1:
                pattern = re.escape(text_to_remove[:-1]) + r"\.?"
            new_df = df.copy()
            cleaned = (
                new_df[col]
                .where(new_df[col].isna(), new_df[col].astype(str).str.replace(pattern, "", regex=True, case=False))
            )
            new_df[col] = cleaned.where(cleaned.isna(), cleaned.astype(str).str.replace(r"\s+", " ", regex=True).str.strip())
            changed = int((df[col].astype(str) != new_df[col].astype(str)).sum())
            if changed == 0:
                return df, f"No '{text_to_remove}' text found in '{col}' — nothing removed."
            return new_df, f"Removed '{text_to_remove}' from {changed:,} value(s) in '{col}'."

    # ---- remove outliers (drop the rows, not cap them) ----
    m = re.search(r"\bremove\s+outliers?\s*(?:in\s+|from\s+)?(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        if not pd.api.types.is_numeric_dtype(df[col]):
            return df, f"'{col}' isn't numeric, so IQR-based outlier removal doesn't apply."
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return df, f"'{col}' has no spread (IQR is 0) — no outliers to remove."
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = len(df)
        new_df = df[(df[col] >= lo) & (df[col] <= hi)].reset_index(drop=True)
        removed = before - len(new_df)
        if removed == 0:
            return df, f"No IQR-based outliers found in '{col}' — nothing removed."
        return new_df, f"Removed {removed:,} outlier row(s) in '{col}' (outside {_fmt(lo)} to {_fmt(hi)})."

    # ---- convert type ----
    m = re.search(r"\b(?:convert|change|make)\s+(?:column\s+)?(.+?)\s+(?:to|into)\s+(?:an?\s+)?(.+)", ql)
    if m:
        col_text, type_phrase_raw = m.group(1), m.group(2)
        col = _match_col(col_text, columns)
        target = _normalize_type_phrase(type_phrase_raw)
        if col and target:
            new_df, note = _convert_type(df, col, target)
            return new_df, note
        if col and not target:
            return df, (
                f"I found column '{col}' but didn't recognise the target type '{type_phrase_raw.strip()}'. "
                "Try: number, integer, text, category, date / time, or boolean."
            )
        # fall through — maybe this wasn't a type-change command after all (e.g. "change" used loosely)


    # ---- filter rows by value (remove / keep only) ----
    m = re.search(r"\b(?:remove|drop)\s+rows?\s+where\s+(.+?)\s+(?:is|=|==|equals?)\s+(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        value = _parse_value_token(m.group(2))
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        mask = df[col].astype(str).str.strip().str.lower() != str(value).strip().lower()
        removed = int((~mask).sum())
        if removed == 0:
            return df, f"No rows found where '{col}' is '{value}' — nothing removed."
        return df[mask].reset_index(drop=True), f"Removed {removed:,} row(s) where '{col}' was '{value}'."

    m = re.search(r"\bkeep\s+only\s+rows?\s+where\s+(.+?)\s+(?:is|=|==|equals?)\s+(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        value = _parse_value_token(m.group(2))
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        mask = df[col].astype(str).str.strip().str.lower() == str(value).strip().lower()
        kept = int(mask.sum())
        if kept == 0:
            return df, f"No rows found where '{col}' is '{value}' — nothing kept (no change made)."
        return df[mask].reset_index(drop=True), f"Kept {kept:,} row(s) where '{col}' is '{value}'; removed the rest."

    # ---- show null rows (read-only, does not modify df) ----
    m = (
        re.search(r"\b(?:show|list|find|display)\s+(?:where\s+)?(?:rows?\s+(?:is|are)\s+)?(?:null|missing|na|empty)\s+(?:in\s+)?(.+)", ql)
        or re.search(r"\b(?:show|list|find|display)\s+(?:null|missing|na|empty)\s+rows?\s+(?:in\s+)?(.+)", ql)
        or re.search(r"\b(?:where|which)\s+(?:rows?\s+(?:is|are)\s+)?(?:null|missing|na|empty)\s+(?:in\s+)?(.+)", ql)
    )
    if m:
        col = _match_col(m.group(1), columns)
        if not col:
            return df, f"I couldn't find a column matching \'{m.group(1).strip()}\'. Available columns: {', '.join(columns)}."
        null_mask = df[col].isna()
        n_null = int(null_mask.sum())
        if n_null == 0:
            return df, f"\'{col}\' has no missing values — all {len(df):,} rows are populated."
        null_indices = df.index[null_mask].tolist()
        row_label = "row" if n_null == 1 else "rows"
        preview_limit = 20
        shown = null_indices[:preview_limit]
        row_list = ", ".join(str(i + 1) for i in shown)
        suffix = f" ... and {n_null - preview_limit} more" if n_null > preview_limit else ""
        pct = round(100 * n_null / len(df), 1)
        msg = (
            f"\'{col}\' has {n_null:,} missing {row_label} ({pct}% of {len(df):,} total).\n"
            f"Row number(s): {row_list}{suffix}.\n"
            f"Tip: use \"fill missing values in {col} with median\" or \"remove rows where {col} is missing\" to fix them."
        )
        return df, msg

    # ---- replace value in column ----
    # Use original-case `q` so that replacements like "replace A1 with X9" preserve casing
    _rep_m = (
        re.search(r"\breplace\s+(.+?)\s+with\s*(.+?)\s+in\s+(.+)", q, re.IGNORECASE)
        or re.search(r"in\s+(.+?)\s+replace\s+(.+?)\s+with\s*(.+)", q, re.IGNORECASE)
    )
    if _rep_m:
        _is_in_first = bool(re.match(r"in\s+", ql))
        if _is_in_first:
            col_text, old_val_raw, new_val_raw = _rep_m.group(1), _rep_m.group(2), _rep_m.group(3)
        else:
            old_val_raw, new_val_raw, col_text = _rep_m.group(1), _rep_m.group(2), _rep_m.group(3)
        col = _match_col(col_text, columns)
        if not col:
            return df, f"I couldn't find a column matching '{col_text.strip()}'. Available columns: {', '.join(columns)}."
        old_val_raw = old_val_raw.strip(" '\"")
        new_val_raw = new_val_raw.strip(" '\"")
        new_df = df.copy()
        col_series = new_df[col]
        # Numeric column — try to coerce both sides to numbers (supports word-numbers like "four")
        if pd.api.types.is_numeric_dtype(col_series):
            old_num = _coerce_numeric(old_val_raw)
            new_num = _coerce_numeric(new_val_raw)
            if old_num is not None and new_num is not None:
                mask = col_series == old_num
                count = int(mask.sum())
                if count == 0:
                    return df, f"No rows in '{col}' equal {old_val_raw} — nothing changed."
                new_df[col] = col_series.where(~mask, new_num)
                return new_df, f"Replaced {count:,} occurrence(s) of {old_val_raw} → {new_val_raw} in '{col}'."
        # String / mixed column — case-insensitive match, preserve new value's casing exactly
        mask = col_series.astype(str).str.strip().str.lower() == old_val_raw.lower()
        count = int(mask.sum())
        if count == 0:
            # Try partial / substring match as fallback
            mask = col_series.astype(str).str.lower().str.contains(re.escape(old_val_raw.lower()), na=False)
            count = int(mask.sum())
            if count == 0:
                return df, f"No rows in '{col}' contain '{old_val_raw}' — nothing changed."
            # Partial: replace the substring, preserve surrounding text
            new_df[col] = col_series.where(
                ~mask,
                col_series.astype(str).str.replace(old_val_raw, new_val_raw, case=False, regex=False)
            )
            return new_df, f"Replaced '{old_val_raw}' → '{new_val_raw}' in {count:,} cell(s) of '{col}' (substring match)."
        new_df[col] = col_series.where(~mask, new_val_raw)
        return new_df, f"Replaced {count:,} occurrence(s) of '{old_val_raw}' → '{new_val_raw}' in '{col}'."

    return df, (
        "I didn't recognise that command. " + HELP_TEXT
    )


# ---------------------------------------------------------------------------
# Bulk "Clean Now" pipeline — applied when the user clicks the Clean Now button
# (as opposed to the per-command chat assistant).  Returns (cleaned_df, log).
# ---------------------------------------------------------------------------

def build_cleaning_recommendations(df: pd.DataFrame) -> dict:
    """Inspect a dataframe and suggest which bulk cleaning options to apply."""
    issues: list[dict] = []

    duplicate_rows = int(df.duplicated().sum())
    duplicate_value_cols = [c for c in df.columns if bool(df[c].duplicated(keep=False).any())]
    if duplicate_rows or duplicate_value_cols:
        issues.append({
            "key": "duplicates",
            "label": "Duplicate rows",
            "detail": (
                f"{duplicate_rows:,} exact duplicate row(s) found."
                if duplicate_rows else
                f"{len(duplicate_value_cols)} column(s) contain repeated values."
            ),
            "columns": duplicate_value_cols,
        })

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        issues.append({
            "key": "constant_columns",
            "label": "Constant columns",
            "detail": f"{len(constant_cols)} column(s) have only one non-missing value.",
            "columns": constant_cols,
        })

    high_missing_cols = [c for c in df.columns if df[c].isna().mean() > 0.5]
    if high_missing_cols:
        issues.append({
            "key": "high_missing_columns",
            "label": "Mostly missing columns",
            "detail": f"{len(high_missing_cols)} column(s) are more than 50% missing.",
            "columns": high_missing_cols,
        })

    missing_numeric = int(df.select_dtypes(include=[np.number]).isna().sum().sum())
    if missing_numeric:
        issues.append({
            "key": "missing_numeric",
            "label": "Missing numeric values",
            "detail": f"{missing_numeric:,} missing numeric value(s) found.",
        })

    cat_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns
    missing_categorical = int(df[cat_cols].isna().sum().sum()) if len(cat_cols) else 0
    if missing_categorical:
        issues.append({
            "key": "missing_categorical",
            "label": "Missing categorical values",
            "detail": f"{missing_categorical:,} missing categorical/text value(s) found.",
        })

    whitespace_cols = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        values = df[col].dropna().astype(str)
        if bool((values != values.str.strip()).any()):
            whitespace_cols.append(col)
    if whitespace_cols:
        issues.append({
            "key": "whitespace",
            "label": "Extra whitespace",
            "detail": f"{len(whitespace_cols)} text column(s) contain leading/trailing spaces.",
            "columns": whitespace_cols,
        })

    mixed_type_cols = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        non_missing = df[col].notna().sum()
        if not non_missing:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = converted.notna().sum() / max(non_missing, 1)
        if valid_ratio >= 0.9:
            mixed_type_cols.append(col)
    if mixed_type_cols:
        issues.append({
            "key": "mixed_types",
            "label": "Numeric values stored as text",
            "detail": f"{len(mixed_type_cols)} text column(s) look numeric.",
            "columns": mixed_type_cols,
        })

    nonstandard_names = [
        c for c in df.columns
        if str(c).strip().lower().replace(" ", "_").replace("-", "_") != c
    ]
    if nonstandard_names:
        issues.append({
            "key": "column_names",
            "label": "Column names can be standardised",
            "detail": f"{len(nonstandard_names)} column name(s) can be normalised.",
            "columns": nonstandard_names,
        })

    recommended_options = {
        "drop_duplicates": bool(duplicate_rows or duplicate_value_cols),
        "fill_missing_numeric": "median" if missing_numeric else "none",
        "fill_missing_categorical": "mode" if missing_categorical else "none",
        "drop_high_missing_cols": bool(high_missing_cols),
        "strip_whitespace": bool(whitespace_cols),
        "drop_constant_cols": bool(constant_cols),
        "fix_mixed_types": bool(mixed_type_cols),
        "fix_column_names": bool(nonstandard_names),
        "remove_outliers": True,
    }

    summary = (
        f"I found {len(issues)} cleaning issue(s) to review."
        if issues else
        "I found no obvious cleaning issues."
    )
    return {
        "summary": summary,
        "issues": issues,
        "recommended_options": recommended_options,
    }


def clean_dataframe(df: pd.DataFrame, options: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply a set of cleaning options to a DataFrame in one shot.
    `options` is a dict matching CleanRequest fields in main.py.
    Returns (cleaned DataFrame, list of human-readable log lines).
    """
    opt = {
        "drop_duplicates": True,
        "fill_missing_numeric": "median",
        "fill_missing_categorical": "mode",
        "drop_high_missing_cols": True,
        "strip_whitespace": True,
        "drop_constant_cols": True,
        "fix_mixed_types": True,
        "fix_column_names": True,
        "remove_outliers": True,
        **(options or {}),
    }
    log: list[str] = []
    out = df.copy()

    # 1. Fix column names
    if opt.get("fix_column_names"):
        renamed = {}
        for col in out.columns:
            new = str(col).strip().lower().replace(" ", "_").replace("-", "_")
            if new and new != col:
                renamed[col] = new
        if renamed:
            out = out.rename(columns=renamed)
            log.append(f"Normalised {len(renamed)} column name(s).")

    # 2. Strip whitespace on object columns
    if opt.get("strip_whitespace"):
        stripped = 0
        for col in out.select_dtypes(include="object").columns:
            before = out[col].astype(str).str.len().sum()
            out[col] = out[col].where(out[col].isna(), out[col].astype(str).str.strip())
            after = out[col].astype(str).str.len().sum()
            if after < before:
                stripped += 1
        if stripped:
            log.append(f"Stripped leading/trailing whitespace from {stripped} column(s).")

    # 3. Fix mixed-type columns (strings that contain numbers)
    if opt.get("fix_mixed_types"):
        fixed = 0
        for col in out.select_dtypes(include="object").columns:
            converted = pd.to_numeric(out[col], errors="coerce")
            valid_ratio = converted.notna().sum() / max(out[col].notna().sum(), 1)
            if valid_ratio >= 0.9:
                out[col] = converted
                fixed += 1
        if fixed:
            log.append(f"Converted {fixed} text column(s) to numeric (≥90 % numeric content).")

    # 4. Drop constant columns
    if opt.get("drop_constant_cols"):
        constant = [c for c in out.columns if out[c].nunique(dropna=True) <= 1]
        if constant:
            out = out.drop(columns=constant)
            log.append(f"Dropped {len(constant)} constant column(s) ({', '.join(constant)}).")

    # 5. Drop high-missing columns
    if opt.get("drop_high_missing_cols"):
        high_missing = [c for c in out.columns if out[c].isna().mean() > 0.5]
        if high_missing:
            out = out.drop(columns=high_missing)
            log.append(f"Dropped {len(high_missing)} column(s) with >50 % missing values.")

    # 6. Drop duplicates
    if opt.get("drop_duplicates"):
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        removed = before - len(out)
        if removed:
            log.append(f"Removed {removed:,} duplicate row(s).")

    # 7. Fill missing numeric values
    num_strategy = opt.get("fill_missing_numeric", "median")
    if num_strategy and num_strategy != "none":
        filled = 0
        for col in out.select_dtypes(include=[np.number]).columns:
            n_missing = int(out[col].isna().sum())
            if n_missing:
                if num_strategy == "mean":
                    val = out[col].mean()
                elif num_strategy == "zero":
                    val = 0
                else:
                    val = out[col].median()
                out[col] = out[col].fillna(val)
                filled += n_missing
        if filled:
            log.append(f"Filled {filled:,} missing numeric value(s) with {num_strategy}.")

    # 8. Fill missing categorical values
    cat_strategy = opt.get("fill_missing_categorical", "mode")
    if cat_strategy and cat_strategy != "none":
        filled = 0
        for col in out.select_dtypes(include="object").columns:
            n_missing = int(out[col].isna().sum())
            if n_missing:
                if cat_strategy == "mode" and not out[col].mode(dropna=True).empty:
                    val = out[col].mode(dropna=True).iloc[0]
                else:
                    val = "Unknown"
                out[col] = out[col].fillna(val)
                filled += n_missing
        if filled:
            log.append(f"Filled {filled:,} missing categorical value(s) with {cat_strategy}.")

    # 9. Remove outliers (capped IQR-based clipping)
    if opt.get("remove_outliers"):
        capped = 0
        for col in out.select_dtypes(include=[np.number]).columns:
            q1 = out[col].quantile(0.25)
            q3 = out[col].quantile(0.75)
            iqr = q3 - q1
            if iqr != 0:
                lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
                before = out[col].isna().sum()
                out[col] = out[col].clip(lo, hi)
                capped += int(out[col].isna().sum() - before)
        if capped:
            log.append(f"Clipped {capped:,} outlier value(s) (3×IQR method).")

    if not log:
        log = ["Clean Now found nothing new to fix — your data is already in good shape."]

    return out, log
