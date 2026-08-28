"""
Chat-based dataset cleaning.

Like assistant.py, this does not call an external AI API — it parses plain
English cleaning instructions with regex/heuristics and applies the matching
pandas operation locally. If a command isn't understood, it explains what
phrasings are supported instead of guessing.
"""
import ast
import operator
import re

import numpy as np
import pandas as pd

from . import nlp
from .assistant import (
    CLEAN_KEYWORDS,
    answer_question_and_table,
    _correct_column_typos,
    _correct_keyword_typos,
    _find_columns_in_text,
    _format_keyword_correction_note,
    _normalise_text,
)


_TYPE_ALIASES = {
    "datetime": {"date/time", "datetime", "date", "time"},
    "integer": {"integer", "int", "wholenumber"},
    "float": {"float", "decimal", "float/decimal", "number", "numeric"},
    "text": {"text", "string", "str"},
    "category": {"category", "categorical"},
    "boolean": {"boolean", "bool", "true/false", "yes/no"},
}

REMOVE_VERB_RE = r"(?:drop|remove|delete|discard|erase|get\s+rid\s+of)"
RENAME_VERB_RE = r"(?:rename|call)"
NAME_JOINER_RE = r"(?:to|into|as)"


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


def _clean_new_column_name(raw: str) -> str:
    text = raw.strip(" '\"?.,:;-")
    text = re.sub(r"^(?:the\s+)?(?:column\s+)?(?:called\s+|named\s+|titled\s+)?", "", text, flags=re.IGNORECASE)
    return text.strip(" '\"?.,:;-")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def _match_col(text: str, columns: list) -> str | None:
    found = _find_columns_in_text(text, columns)
    return found[0] if found else None


def _only_empty_column(df: pd.DataFrame) -> str | None:
    empty_cols = [col for col in df.columns if bool(df[col].isna().all())]
    return empty_cols[0] if len(empty_cols) == 1 else None


def _default_text_column(df: pd.DataFrame) -> str | None:
    text_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return text_cols[0] if len(text_cols) == 1 else None


def _split_first_word_columns(
    df: pd.DataFrame,
    source_col: str,
    first_col: str,
    rest_col: str,
) -> pd.DataFrame:
    out = df.copy()
    cleaned = out[source_col].astype("string").str.strip()
    parts = cleaned.str.extract(r"^(\S+)(?:\s+(.*))?$")
    blank_or_missing = cleaned.isna() | cleaned.eq("")

    out[first_col] = parts[0].mask(blank_or_missing, pd.NA)
    out[rest_col] = parts[1].fillna("").mask(blank_or_missing, pd.NA)
    return out


# Safe mathematical expression evaluator
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_math(expr: str, df: pd.DataFrame, row: pd.Series = None) -> float | None:
    """Safely evaluate a mathematical expression with column references.
    
    Supports: +, -, *, /, //, %, **, parentheses
    Column names are replaced with their values from the current row.
    """
    try:
        # Replace column names with their values
        eval_expr = expr
        if row is not None:
            # Sort by length (longest first) to avoid partial replacements
            cols_sorted = sorted(df.columns, key=len, reverse=True)
            for col in cols_sorted:
                # Use case-insensitive matching for column names
                col_pattern = r'\b' + re.escape(str(col)) + r'\b'
                if re.search(col_pattern, eval_expr, re.IGNORECASE):
                    val = row[col]
                    if pd.isna(val):
                        return np.nan
                    eval_expr = re.sub(col_pattern, f"({val})", eval_expr, flags=re.IGNORECASE)
        
        # Parse and evaluate
        tree = ast.parse(eval_expr, mode='eval')
        
        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            elif isinstance(node, ast.Constant):  # Python 3.8+
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant: {node.value}")
            # Note: ast.Num (pre-3.8) is intentionally not handled — every numeric
            # literal on supported Python versions parses as ast.Constant above,
            # and ast.Num itself is removed in Python 3.14, so referencing it here
            # would be both dead code and a future crash.
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op_type = type(node.op)
                if op_type in _SAFE_OPERATORS:
                    return _SAFE_OPERATORS[op_type](left, right)
                raise ValueError(f"Unsupported operator: {op_type}")
            elif isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op_type = type(node.op)
                if op_type in _SAFE_OPERATORS:
                    return _SAFE_OPERATORS[op_type](operand)
                raise ValueError(f"Unsupported unary operator: {op_type}")
            elif isinstance(node, ast.Call):
                raise ValueError("Function calls are not allowed")
            else:
                raise ValueError(f"Unsupported expression: {type(node)}")
        
        return _eval(tree)
    except Exception:
        return None


def _apply_aggregate_operation(df: pd.DataFrame, operation: str, columns: list = None) -> pd.Series:
    """Apply an aggregate operation across columns.
    
    Args:
        df: DataFrame
        operation: 'sum', 'mean', 'average', 'min', 'max', 'product', 'multiply'
        columns: list of columns to operate on (None = all numeric columns)
    
    Returns:
        Series with the result for each row
    """
    if columns is None:
        # Use all numeric columns by default
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columns:
        return pd.Series([np.nan] * len(df))
    
    # Filter to only columns that exist in df
    columns = [c for c in columns if c in df.columns]
    
    if not columns:
        return pd.Series([np.nan] * len(df))
    
    # Get the data subset
    data = df[columns]
    
    # Apply operation across columns (axis=1)
    if operation in ('sum', 'add', 'total'):
        return data.sum(axis=1, skipna=True)
    elif operation in ('mean', 'average', 'avg'):
        return data.mean(axis=1, skipna=True)
    elif operation == 'min':
        return data.min(axis=1, skipna=True)
    elif operation == 'max':
        return data.max(axis=1, skipna=True)
    elif operation in ('product', 'multiply'):
        return data.product(axis=1, skipna=True)
    elif operation == 'median':
        return data.median(axis=1, skipna=True)
    elif operation == 'std':
        return data.std(axis=1, skipna=True)
    else:
        return pd.Series([np.nan] * len(df))


def _parse_math_expression(text: str, df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Parse a mathematical expression from text like 'price * 1.2' or 'quantity + 10'.
    Returns (expression, new_column_name) or (None, None) if not found.
    """
    # Pattern: "create column X as expression" or "X = expression"
    # or "expression as X" or just the expression
    
    # Try to find explicit column name with various separators
    create_patterns = [
        r"create\s+(?:a\s+)?(?:new\s+)?column\s+(?:called\s+|named\s+)?(\w+)\s+(?:as|with|from|=)\s+(.+)",
        r"add\s+(?:a\s+)?(?:new\s+)?column\s+(?:called\s+|named\s+)?(\w+)\s+(?:as|with|from|=)\s+(.+)",
        r"new\s+column\s+(?:called\s+|named\s+)?(\w+)\s+(?:as|with|from|=)\s+(.+)",
        r"column\s+(?:called\s+|named\s+)?(\w+)\s*=\s*(.+)",
        # Handle "where it is" or "where" patterns
        r"create\s+(?:a\s+)?(?:new\s+)?column\s+(?:called\s+|named\s+)?(\w+)\s+where\s+(?:it\s+)?is\s+(.+)",
        r"add\s+(?:a\s+)?(?:new\s+)?column\s+(?:called\s+|named\s+)?(\w+)\s+where\s+(?:it\s+)?is\s+(.+)",
        r"new\s+column\s+(?:called\s+|named\s+)?(\w+)\s+where\s+(?:it\s+)?is\s+(.+)",
    ]
    
    for pattern in create_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            col_name = m.group(1).strip()
            expr = m.group(2).strip()
            return expr, col_name
    
    # Try "expression as column_name"
    as_pattern = r"(.+?)\s+as\s+(\w+)$"
    m = re.search(as_pattern, text, re.IGNORECASE)
    if m:
        expr = m.group(1).strip()
        col_name = m.group(2).strip()
        return expr, col_name
    
    return None, None


def _parse_empty_column_name(text: str) -> str | None:
    patterns = [
        r"\b(?:create|add|make)\s+(?:a\s+)?(?:new\s+)?(?:empty|blank)?\s*column\s+(?:called\s+|named\s+)?(\w+)\s*$",
        r"\b(?:create|add|make)\s+(?:a\s+)?(?:empty|blank)\s+(?:new\s+)?column\s+(?:called\s+|named\s+)?(\w+)\s*$",
        r"\bnew\s+(?:empty|blank)?\s*column\s+(?:called\s+|named\s+)?(\w+)\s*$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip(" '\"?.,:;-")
    return None


def _parse_fill_column_expression(text: str) -> tuple[str | None, str | None]:
    patterns = [
        r"\b(?:fill|populate|update)\s+(?:the\s+)?(?:column\s+)?(.+?)\s+(?:with|as|from|=)\s+(.+)",
        r"\b(?:set|calculate|compute)\s+(?:the\s+)?(?:column\s+)?(.+?)\s+(?:to|as|with|from|=)\s+(.+)",
        r"\b(?:fill|populate|update)\s+(?:it|the\s+new\s+column|the\s+empty\s+column|the\s+blank\s+column)\s+(?:with|as|from|=)\s+(.+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        if len(m.groups()) == 1:
            return m.group(1).strip(), "it"
        return m.group(2).strip(), m.group(1).strip()
    return None, None


def _apply_math_expression_to_column(
    df: pd.DataFrame,
    target_col: str,
    expr: str,
    action_word: str,
) -> tuple[pd.DataFrame, str]:
    if df.empty:
        new_df = df.copy()
        new_df[target_col] = pd.Series(dtype="float64")
        return new_df, f"{action_word} '{target_col}' with expression: {expr}"

    test_row = df.dropna().iloc[0] if len(df.dropna()) > 0 else df.iloc[0]
    test_result = _safe_eval_math(expr, df, test_row)
    if test_result is None:
        return df, f"Could not evaluate the expression '{expr}'. Make sure it uses valid column names and operators (+, -, *, /, **, %)."

    new_df = df.copy()
    try:
        new_df[target_col] = df.apply(lambda row: _safe_eval_math(expr, df, row), axis=1)
        return new_df, f"{action_word} '{target_col}' with expression: {expr}"
    except Exception as e:
        return df, f"Error filling column: {e}"


def _match_two_cols(text: str, columns: list) -> tuple[str | None, str | None]:
    found = _find_columns_in_text(text, columns)
    if len(found) >= 2:
        return found[0], found[1]
    return (found[0], None) if found else (None, None)


def _parse_value_token(raw: str):
    raw = raw.strip(" '\"?.,:;-")
    # Word number mapping
    word_numbers = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
    }
    
    # Check if it's a word number
    if raw.lower() in word_numbers:
        return word_numbers[raw.lower()]
    
    # Try numeric parsing
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
    "• remove duplicates\n"
    "• remove duplicates of column age\n"
    "• drop column notes  (also: remove / delete / discard / erase column notes)\n"
    "• rename column dob to date_of_birth  (also: rename dob into date_of_birth / rename dob as date_of_birth)\n"
    "• create a new column called revenue  (creates an empty column)\n"
    "• fill revenue with price * quantity  (also: populate / update / set)\n"
    "• create column total as price * quantity  (math operations)\n"
    "• add new column tax as price * 0.15\n"
    "• new column profit = revenue - cost\n"
    "• create column doubled as quantity * 2\n"
    "• split first word from full_name into title and name\n"
    "• replace 2 with 0 in profit  (replaces exact values in a column)\n"
    "• replace yes with 1 in discount_applied\n"
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
    "• check distinct status values and counts\n"
    "• show default rate by grade\n"
    "• average revenue and profit by region\n"
    "• convert price to number  (or: integer / text / category / date / time / boolean)\n"
    "• change ORDER_DATE to date / time\n"
    "• round price to 2 decimal places  (also: round latitude to 2 dp / keep 2 decimals in longitude / round all numeric columns to 3 decimals)\n"
    "• reset  (undoes every chat cleaning command and restores the original upload)\n"
    "\n"
    "Math operations: +, -, *, /, ** (power), % (modulo)\n"
    "Examples: 'price * 1.2', 'quantity + 10', '(price - cost) / cost * 100'"
)


def run_command(df: pd.DataFrame, text: str, original_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    new_df, message, _ = run_command_with_table(df, text, original_df)
    return new_df, message


def run_command_with_table(
    df: pd.DataFrame,
    text: str,
    original_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, dict | None]:
    corrected_text, corrections = _correct_keyword_typos(text.strip(), CLEAN_KEYWORDS)
    new_df, message, table = _run_command_impl_with_table(df, corrected_text, original_df)
    return new_df, _format_keyword_correction_note(corrections) + message, table


def run_database_analysis(
    df: pd.DataFrame,
    question: str,
    bin_column: str | None = None,
    bin_edges: list | None = None,
    bin_labels: list | None = None,
) -> tuple[str, dict | None]:
    """Backs the "Group & Aggregate" panel in the Data Cleaning tab: a
    plain-English grouped/aggregated question, optionally over a numeric
    column bucketed into custom ranges first (e.g. income into "Under 40k",
    "40k-80k", ... — the same idea as a SQL "CASE WHEN ... END" bucketing
    column). Raises ValueError with a user-facing message on bad input;
    the caller (the API layer) is expected to turn that into an HTTP error.
    """
    if bin_column:
        if bin_column not in df.columns:
            raise ValueError(f"Column '{bin_column}' was not found.")
        if not pd.api.types.is_numeric_dtype(df[bin_column]):
            raise ValueError(f"'{bin_column}' is not numeric, so it can't be bucketed into ranges.")
        if not bin_edges or not bin_labels:
            raise ValueError("Provide range breakpoints and labels to bucket a numeric column.")
        if len(bin_labels) != len(bin_edges) + 1:
            raise ValueError("The number of bucket labels must be exactly one more than the number of breakpoints.")

        band_col = f"{bin_column}_band"
        bins = [-float("inf")] + list(bin_edges) + [float("inf")]
        # right=False -> half-open [a, b) intervals, matching "annual_inc < 40000" style
        # SQL CASE WHEN bucketing (each breakpoint is an exclusive upper bound).
        banded = pd.cut(df[bin_column], bins=bins, labels=bin_labels, right=False, ordered=True)
        df = df.copy()
        df[band_col] = banded
        # Drop the raw numeric column from the working copy: its name (e.g. "annual_inc")
        # is a text substring of the new band column ("annual_inc_band"), which would
        # otherwise make the question-parser mistakenly treat it as also mentioned and
        # pull it into any numeric aggregation alongside the intended metric column(s).
        df = df.drop(columns=[bin_column])

        if band_col not in question:
            raise ValueError("The question must reference the bucketed column.")

    return answer_question_and_table(df, question)


def _looks_like_analysis_question(text: str) -> bool:
    q = _normalise_text(text)
    tokens = set(q.split())
    analysis_terms = {
        "average", "avg", "mean", "median", "sum", "total", "rate", "percentage",
        "percent", "pct", "distinct", "count", "counts", "frequency", "frequencies",
        "value", "values",
    }
    if not tokens.intersection(analysis_terms):
        return False
    return bool(
        tokens.intersection({"distinct", "frequency", "frequencies"})
        or any(term in q for term in (" by ", " per ", " across ", " group ", " grouped ", " value count", " value counts"))
    )


def _run_command_impl_with_table(
    df: pd.DataFrame,
    text: str,
    original_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, dict | None]:
    if _looks_like_analysis_question(text):
        # Single pass: compute the answer text and its table together instead of
        # calling answer_question() and answer_question_table() separately, which
        # would redundantly re-parse the question and re-run the same group-by twice.
        answer, table = answer_question_and_table(df, text)
        if table:
            return df, answer, table
    new_df, message = _run_command_impl(df, text, original_df)
    return new_df, message, None


def _run_command_impl(df: pd.DataFrame, text: str, original_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
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
    if re.search(r"\b(reset|undo|restore|start over|revert)\b", ql):
        if original_df is None:
            return df, "No original data is available to reset to."
        return original_df.copy(), "Restored the original uploaded data — every chat cleaning command has been undone."

    # ---- round numeric column(s) to N decimal places ----
    # "round price to 2 decimal places" / "round latitude to 2 dp" /
    # "keep 2 decimals in longitude" / "round all numeric columns to 3 decimals"
    _round_m = (
        re.search(r"\bround\s+(?:off\s+|up\s+|down\s+)?(?:the\s+)?(?:column\s+)?(.+?)\s+(?:to|at|with)\s+(\d+)\s*(?:decimal|dec|dp|d\.p\.?|places?|digits?)", ql)
        or re.search(r"\bround\s+(?:the\s+)?(?:column\s+)?(.+?)\s+(?:to|at)\s+(\d+)\b", ql)
        or re.search(r"\b(?:keep|show|limit|reduce|set|use|leave)\s+(?:only\s+)?(\d+)\s*(?:decimal|dec|dp|d\.p\.?|places?|digits?)[a-z\s]*?\s+(?:in|for|on|of)\s+(.+)", ql)
        or re.search(r"\b(\d+)\s*(?:decimal\s*places?|decimals?|dp)\b\s+(?:in|for|on|of)\s+(.+)", ql)
    )
    if _round_m:
        g1, g2 = _round_m.group(1).strip(), _round_m.group(2).strip()
        if g1.isdigit():                       # "keep 2 decimals in price"
            decimals, col_text = int(g1), g2
        else:                                  # "round price to 2 decimal places"
            decimals, col_text = int(g2), g1

        if decimals > 15:
            return df, "Please pick a number of decimal places between 0 and 15."

        col_key = re.sub(r"^(?:the\s+|all\s+)", "", col_text).strip(" '\"?.,:;-")
        all_scope = bool(re.fullmatch(r"(?:all\s+)?(?:numeric|number|numerical|float|decimal)?\s*columns?|everything|all", col_key)) \
            or col_key in ("all", "everything", "all columns", "numeric columns")

        new_df = df.copy()
        if all_scope:
            targets = new_df.select_dtypes(include="number").columns.tolist()
            if not targets:
                return df, "There are no numeric columns to round."
        else:
            col = _match_col(col_text, columns)
            if not col:
                return df, f"I couldn't find a column matching '{col_text}'. Available columns: {', '.join(columns)}."
            targets = [col]

        rounded, skipped = [], []
        for col in targets:
            series = pd.to_numeric(new_df[col], errors="coerce")
            if series.notna().sum() == 0:
                skipped.append(col)
                continue
            series = series.round(decimals)
            new_df[col] = series.astype("Int64") if decimals == 0 else series
            rounded.append(col)

        if not rounded:
            return df, f"'{targets[0]}' isn't numeric, so it can't be rounded. Convert it to a number first."

        unit = "whole number" if decimals == 0 else f"{decimals} decimal place{'s' if decimals != 1 else ''}"
        msg = f"Rounded {', '.join(chr(39) + c + chr(39) for c in rounded)} to {unit}."
        if skipped:
            msg += f" Skipped (not numeric): {', '.join(skipped)}."
        return new_df, msg


    # ---- split first word into two new columns ----
    split_patterns = [
        r"\b(?:split|separate|extract)\s+(?:the\s+)?first\s+word\s+(?:from|in|of)\s+(.+?)\s+(?:into|to)\s+(?:columns?\s+)?(\w+)\s+(?:and|,)\s+(\w+)\b",
        r"\b(?:split|separate|extract)\s+(.+?)\s+(?:by|on|using)\s+(?:the\s+)?first\s+word\s+(?:into|to)\s+(?:columns?\s+)?(\w+)\s+(?:and|,)\s+(\w+)\b",
        r"\bcreate\s+(?:new\s+)?columns?\s+(\w+)\s+(?:and|,)\s+(\w+)\s+from\s+(.+?)\s+(?:by\s+)?(?:splitting|separating|extracting)\s+(?:the\s+)?first\s+word\b",
    ]
    for i, pattern in enumerate(split_patterns):
        m = re.search(pattern, q, re.IGNORECASE)
        if not m:
            continue

        if i == 2:
            first_col, rest_col, source_text = m.group(1), m.group(2), m.group(3)
        else:
            source_text, first_col, rest_col = m.group(1), m.group(2), m.group(3)

        source_key = re.sub(r"^(?:the|a|an)\s+", "", source_text.strip().lower())
        source_col = _match_col(source_text, columns)
        if not source_col and source_key in ("row", "rows", "cell", "value", "values"):
            source_col = _default_text_column(df)
        if not source_col:
            return df, (
                f"I couldn't find the source column to split. Available columns: {', '.join(columns)}. "
                "Try: split first word from full_name into title and name."
            )
        if first_col in columns or rest_col in columns:
            return df, f"'{first_col}' or '{rest_col}' already exists. Choose new column names."
        if first_col == rest_col:
            return df, "Use two different names for the new columns."

        new_df = _split_first_word_columns(df, source_col, first_col, rest_col)
        return new_df, f"Split first word from '{source_col}' into '{first_col}' and '{rest_col}'."

    # ---- create new column with math expression ----
    if any(term in ql for term in ("create", "add", "new column")):
        # Pull an optional "before X" / "after X" clause out before any of the
        # patterns below run, so it doesn't get swallowed as part of an
        # expression or column name. Also treat "titled" the same as
        # "called"/"named" (all the patterns below already understand those),
        # and fix squashed/typo'd column references (e.g. "unitprice" for
        # "unit price") the same forgiving way the rest of the assistant does.
        position = None
        position_col_raw = None
        pos_match = re.search(r"\b(before|after)\b\s+(.+)$", q, flags=re.IGNORECASE)
        if pos_match:
            position = pos_match.group(1).lower()
            position_col_raw = pos_match.group(2).strip(" '\"?.,:;-")
            q = q[:pos_match.start()].strip()
        q = re.sub(r"\btitled\b", "named", q, flags=re.IGNORECASE)
        q = _correct_column_typos(q, columns)
        ql = q.lower()

        position_col = None
        if position_col_raw:
            corrected_ref = _correct_column_typos(position_col_raw, columns)
            for col in sorted(columns, key=lambda c: len(str(c)), reverse=True):
                if re.search(rf"\b{re.escape(str(col))}\b", corrected_ref, flags=re.IGNORECASE):
                    position_col = col
                    break

        def _place(new_df: pd.DataFrame, new_col: str, message: str) -> tuple[pd.DataFrame, str]:
            """Applies the before/after positioning captured above, if any,
            then returns the (df, message) pair a caller should return."""
            if position_col and position_col in new_df.columns and new_col in new_df.columns:
                ordered = [c for c in new_df.columns if c != new_col]
                idx = ordered.index(position_col)
                insert_at = idx if position == "before" else idx + 1
                ordered.insert(insert_at, new_col)
                new_df = new_df[ordered]
                message += f" ({position} '{position_col}')"
            elif position_col_raw and not position_col:
                message += " — couldn't find that reference column, so it was added at the end"
            return new_df, message

        empty_col_name = _parse_empty_column_name(q)
        if empty_col_name:
            if empty_col_name in columns:
                return df, f"Column '{empty_col_name}' already exists."
            new_df = df.copy()
            new_df[empty_col_name] = pd.NA
            return _place(new_df, empty_col_name, f"Created empty column '{empty_col_name}'.")

        # First check if this is an aggregate operation (sum/mean/multiply all columns)
        aggregate_match = re.search(
            r"(?:create|add|new)\s+(?:a\s+)?(?:new\s+)?column\s+(?:called\s+|named\s+)?(\w+)\s+"
            r"(?:as|with|from|=|where\s+(?:it\s+)?is\s+)?\s*"
            r"(sum|add|total|mean|average|multiply|product|min|max|median)\s+(?:of\s+)?(?:all\s+)?(?:columns?)?",
            ql
        )
        
        if aggregate_match:
            new_col_name = aggregate_match.group(1)
            operation = aggregate_match.group(2)

            if new_col_name in columns:
                return df, f"Column '{new_col_name}' already exists. Choose a different name."

            # Apply the aggregate operation
            result_series = _apply_aggregate_operation(df, operation)
            
            if result_series.isna().all():
                return df, f"Could not perform {operation} operation. Make sure you have numeric columns."
            
            new_df = df.copy()
            new_df[new_col_name] = result_series
            
            op_name = {
                'sum': 'sum', 'add': 'sum', 'total': 'sum',
                'mean': 'mean', 'average': 'mean', 'avg': 'mean',
                'multiply': 'product', 'product': 'product',
                'min': 'min', 'max': 'max', 'median': 'median'
            }.get(operation, operation)
            
            return _place(new_df, new_col_name, f"Created new column '{new_col_name}' as {op_name} of all numeric columns")
        
        # Otherwise, try to parse as a regular expression
        expr, new_col_name = _parse_math_expression(q, df)
        
        if expr and new_col_name:
            if new_col_name in columns:
                return df, f"Column '{new_col_name}' already exists. Use: fill {new_col_name} with {expr}"
            new_df, message = _apply_math_expression_to_column(df, new_col_name, expr, "Created new column")
            return _place(new_df, new_col_name, message)
        else:
            return df, (
                "To create a new column, use formats like:\n"
                "• create a new column called revenue  (empty column)\n"
                "• fill revenue with price * quantity\n"
                "• create column total as price * quantity\n"
                "• add new column tax as price * 0.15\n"
                "• new column profit = revenue - cost\n"
                "• create column sum_all as sum of all columns\n"
                "• create column avg as mean of all columns\n"
                "• create column total as multiply all columns\n"
                "• create a new column named revenue before/after cost  (position it)\n"
                "Supports: +, -, *, /, ** (power), % (modulo)\n"
                "You can use any existing column names in the expression."
            )

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
    m = re.search(rf"\b{REMOVE_VERB_RE}\s+(?:the\s+)?column\s+(.+)", ql) or \
        re.search(rf"\b{REMOVE_VERB_RE}\s+(.+?)\s+column\b", ql) or \
        re.search(rf"column\s+(.+?)\s+{REMOVE_VERB_RE}\b", ql)
    if m:
        col = _match_col(m.group(1), columns) or _match_col(q, columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        return df.drop(columns=[col]), f"Dropped column '{col}'."

    # ---- decimal separator fix: replace , with . (European/African number format) ----
    # MUST be before the rename handler — "change comma to dot in price" is otherwise
    # caught by the rename pattern "change <col> to <name>".
    # Handles all natural phrasings:
    #   "replace , with . in price"           → convert price to float
    #   "replace , with . in all columns"     → all eligible columns
    #   "replace comma with dot in price"     → convert price to float
    #   "fix decimal separator in price"      → convert price to float
    #   "change comma to dot in revenue"      → convert revenue to float
    #   "convert comma decimal to dot"        → all eligible columns
    _is_comma_to_dot = (
        re.search(r"\breplace\s+,\s+with\s+\.", ql)
        or re.search(r"\breplace\s+comma\s+with\s+(?:dot|period|point)", ql)
        or re.search(r"\bfix\s+decimal\s+separator\b", ql)
        or re.search(r"\bconvert\s+(?:comma|,)\s+decimal", ql)
        or re.search(r"\bdecimal\s+(?:separator|delimiter)\b", ql)
        or re.search(r"\bchange\s+(?:comma|,)\s+(?:decimal\s+)?to\s+(?:dot|period|point|\.)\b", ql)
        or re.search(r"\bchange\s+comma\s+to\s+dot\b", ql)
    )

    if _is_comma_to_dot:
        _all_scope = bool(
            re.search(r"\ball\s+(?:numeric\s+)?columns?\b", ql)
            or re.search(r"\beverything\b", ql)
            or not re.search(r"\bin\s+\w", ql)
        )
        if _all_scope:
            candidate_cols = columns
        else:
            _mentioned = _match_col(ql, columns)
            candidate_cols = [_mentioned] if _mentioned else columns

        new_df = df.copy()
        converted, skipped = [], []

        for col in candidate_cols:
            series = new_df[col]
            if pd.api.types.is_numeric_dtype(series):
                skipped.append(col)
                continue
            s_str = series.astype(str)
            has_comma_decimal = s_str.str.contains(r'\d,\d', regex=True, na=False).any()
            if not has_comma_decimal:
                skipped.append(col)
                continue
            # Strip thousand-dot separators first (e.g. "1.234,56" → "1234,56")
            cleaned = s_str.str.replace(r'(?<=\d)\.(?=\d{3}(?:[,\s]|$))', '', regex=True)
            cleaned = cleaned.str.replace(',', '.', regex=False)
            numeric = pd.to_numeric(cleaned, errors='coerce')
            before_valid = int(series.notna().sum())
            after_valid  = int(numeric.notna().sum())
            if after_valid >= max(before_valid * 0.7, 1):
                new_df[col] = numeric
                converted.append(col)
            else:
                skipped.append(col)

        if not converted:
            return df, (
                "No columns appeared to contain comma-decimal numbers (like '1,23' or '1.234,56'). "
                "Make sure the column is stored as text, not already numeric."
            )
        parts = [f"'{c}'" for c in converted]
        msg = f"Replaced comma decimals with dots and converted to numeric: {', '.join(parts)}."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)} (already numeric or no comma-decimals found)."
        return new_df, msg

    # ---- rename column ----
    m = re.search(rf"\b{RENAME_VERB_RE}\s+(?:the\s+)?(?:column\s+)?(.+?)\s+{NAME_JOINER_RE}\s+(.+)", ql) or \
        re.search(rf"change\s+(?:the\s+)?(?:name\s+of\s+)?(?:column\s+)?(.+?)\s+{NAME_JOINER_RE}\s+(.+)", ql)
    if m:
        old_col = _match_col(m.group(1), columns)
        new_name = _clean_new_column_name(m.group(2))
        if not old_col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        if not new_name:
            return df, "Tell me the new name, for example: rename dob to date_of_birth."
        return df.rename(columns={old_col: new_name}), f"Renamed column '{old_col}' to '{new_name}'."

    # ---- fill/update an existing column with a math expression ----
    if (
        re.search(r"\b(?:fill|populate|update|set|calculate|compute)\b", ql)
        and not re.search(r"\b(?:fill|populate|impute|replace\s+missing)\s+(?:missing|na|null|empty)\b", ql)
    ):
        expr, target_text = _parse_fill_column_expression(q)
        if expr and target_text:
            target_key = target_text.strip().lower()
            if target_key in ("it", "new column", "empty column", "blank column", "the new column"):
                target_col = _only_empty_column(df)
                if not target_col:
                    return df, "Tell me which column to fill, for example: fill revenue with price * quantity."
            else:
                target_col = _match_col(target_text, columns)
                if not target_col:
                    return df, f"I couldn't find a column matching '{target_text}'. Available columns: {', '.join(columns)}."
            return _apply_math_expression_to_column(df, target_col, expr, "Filled column")

    # ---- fill missing ----
    m = re.search(r"\bfill\s+(?:missing|na|null|empty)\s*(?:values?)?\s*(?:in\s+)?(.+?)\s+with\s+(.+)", ql) or \
        re.search(r"fill\s+(.+?)\s+(?:missing|na|null|empty)\s*values?\s*with\s+(.+)", ql) or \
        re.search(r"(?:impute|populate|replace\s+missing)\s*(?:values?\s*)?(?:in\s+)?(.+?)\s+with\s+(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        strategy_raw = m.group(2).strip(" '\"?.,:;-")
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            return df, f"'{col}' has no missing values — nothing to fill."
        new_df = df.copy()
        if strategy_raw in ("mean", "average") and pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].mean()
        elif strategy_raw == "median" and pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].median()
        elif strategy_raw == "mode":
            mode = df[col].mode(dropna=True)
            val = mode.iloc[0] if len(mode) else "Unknown"
        elif strategy_raw in ("zero", "0"):
            val = 0
        else:
            val = _parse_value_token(strategy_raw)
        new_df[col] = new_df[col].fillna(val)
        return new_df, f"Filled {n_missing:,} missing value(s) in '{col}' with {_fmt(val)}."

    # ---- drop rows where col is missing ----
    m = re.search(rf"\b{REMOVE_VERB_RE}\s+rows?\s+where\s+(.+?)\s+is\s+(?:missing|null|na|empty)\b", ql) or \
        re.search(rf"\b{REMOVE_VERB_RE}\s+(?:rows?\s+with\s+)?(?:missing|null|empty)\s+(?:values?\s+in\s+)?(.+)", ql) or \
        re.search(r"delete\s+(?:rows?\s+)?where\s+(.+?)\s+(?:is\s+)?(?:missing|null|na|empty)", ql)
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
    m = re.search(r"\b(lowercase|uppercase|lower|upper)\s+(?:column\s+)?(.+)", ql) or \
        re.search(r"\bmake\s+(.+?)\s+(lowercase|uppercase|lower|upper)\b", ql) or \
        re.search(r"convert\s+(.+?)\s+(?:to\s+)?(lower|upper)\s*case", ql)
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

    # ---- filter rows by comparison operators (less than, greater than, etc.) ----
    # IMPORTANT: This comes BEFORE "remove exact text" to avoid conflicts
    comparison_patterns = [
        # Pattern: "remove rows where col < value" or "remove rows where col is less than value"
        rf"\b{REMOVE_VERB_RE}\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:less\s+than\s+or\s+equal\s+to|<=)\s+(.+)",
        rf"\b{REMOVE_VERB_RE}\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:greater\s+than\s+or\s+equal\s+to|>=)\s+(.+)",
        rf"\b{REMOVE_VERB_RE}\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:less\s+than|<)\s+(.+)",
        rf"\b{REMOVE_VERB_RE}\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:greater\s+than|>)\s+(.+)",
        # Pattern: "remove rows in col where values is less than value"
        rf"\b{REMOVE_VERB_RE}\s+rows?\s+(?:in|from)\s+(.+?)\s+where\s+(?:values?\s+)?(?:is\s+)?(?:less\s+than\s+or\s+equal\s+to|<=)\s+(.+)",
        rf"\b{REMOVE_VERB_RE}\s+rows?\s+(?:in|from)\s+(.+?)\s+where\s+(?:values?\s+)?(?:is\s+)?(?:greater\s+than\s+or\s+equal\s+to|>=)\s+(.+)",
        rf"\b{REMOVE_VERB_RE}\s+rows?\s+(?:in|from)\s+(.+?)\s+where\s+(?:values?\s+)?(?:is\s+)?(?:less\s+than|<)\s+(.+)",
        rf"\b{REMOVE_VERB_RE}\s+rows?\s+(?:in|from)\s+(.+?)\s+where\s+(?:values?\s+)?(?:is\s+)?(?:greater\s+than|>)\s+(.+)",
        # Pattern: "remove less than 0 in quantity" or "remove greater than 100 from price"
        rf"\b{REMOVE_VERB_RE}\s+(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:less\s+than\s+or\s+equal\s+to|<=)\s+(.+?)\s+(?:in|from)\s+(.+)",
        rf"\b{REMOVE_VERB_RE}\s+(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:greater\s+than\s+or\s+equal\s+to|>=)\s+(.+?)\s+(?:in|from)\s+(.+)",
        rf"\b{REMOVE_VERB_RE}\s+(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:less\s+than|<)\s+(.+?)\s+(?:in|from)\s+(.+)",
        rf"\b{REMOVE_VERB_RE}\s+(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:greater\s+than|>)\s+(.+?)\s+(?:in|from)\s+(.+)",
    ]
    
    for pattern in comparison_patterns:
        m = re.search(pattern, ql)
        if m:
            # Determine which group is column and which is value
            first_group = m.group(1).strip()
            second_group = m.group(2).strip()
            
            # Try to parse as number (use _parse_value_token to handle word numbers like "zero")
            first_parsed = _parse_value_token(first_group)
            second_parsed = _parse_value_token(second_group)
            first_is_num = isinstance(first_parsed, (int, float))
            second_is_num = isinstance(second_parsed, (int, float))
            
            if first_is_num and not second_is_num:
                # Pattern: "remove less than 0 in quantity"
                value = first_parsed
                col_text = second_group
            else:
                # Pattern: "remove rows where quantity < 0"
                col_text = first_group
                value = second_parsed
            
            col = _match_col(col_text, columns)
            if not col:
                return df, f"I couldn't find a column matching '{col_text.strip()}'. Available columns: {', '.join(columns)}."
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                return df, f"'{col}' isn't numeric, so comparison operators don't apply."
            
            # Determine operator
            operator = None
            full_match = m.group(0)
            if re.search(r"less\s+than\s+or\s+equal\s+to|<=|<=", full_match):
                operator = "<="
            elif re.search(r"greater\s+than\s+or\s+equal\s+to|>=|>=", full_match):
                operator = ">="
            elif re.search(r"less\s+than|<(?!\s*=|>)", full_match):
                operator = "<"
            elif re.search(r"greater\s+than|>(?!\s*=|>)", full_match):
                operator = ">"
            
            before = len(df)
            if operator == "<":
                new_df = df[df[col] >= value].reset_index(drop=True)
                op_text = "less than"
            elif operator == ">":
                new_df = df[df[col] <= value].reset_index(drop=True)
                op_text = "greater than"
            elif operator == "<=":
                new_df = df[df[col] > value].reset_index(drop=True)
                op_text = "less than or equal to"
            elif operator == ">=":
                new_df = df[df[col] < value].reset_index(drop=True)
                op_text = "greater than or equal to"
            
            removed = before - len(new_df)
            if removed == 0:
                return df, f"No rows found where '{col}' is {op_text} {value} — nothing removed."
            return new_df, f"Removed {removed:,} row(s) where '{col}' was {op_text} {value}."
    
    # ---- keep only rows by comparison operators ----
    keep_patterns = [
        r"\bkeep\s+only\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:less\s+than|<)\s+(.+)",
        r"\bkeep\s+only\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:greater\s+than|>)\s+(.+)",
        r"\bkeep\s+only\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:less\s+than\s+or\s+equal\s+to|<=)\s+(.+)",
        r"\bkeep\s+only\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:greater\s+than\s+or\s+equal\s+to|>=)\s+(.+)",
        r"\bkeep\s+(?:only\s+)?(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:less\s+than|<)\s+(.+?)\s+(?:in|from)\s+(.+)",
        r"\bkeep\s+(?:only\s+)?(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:greater\s+than|>)\s+(.+?)\s+(?:in|from)\s+(.+)",
    ]
    
    for pattern in keep_patterns:
        m = re.search(pattern, ql)
        if m:
            first_group = m.group(1).strip()
            second_group = m.group(2).strip()
            
            # Try to parse as number (use _parse_value_token to handle word numbers like "zero")
            first_parsed = _parse_value_token(first_group)
            second_parsed = _parse_value_token(second_group)
            first_is_num = isinstance(first_parsed, (int, float))
            second_is_num = isinstance(second_parsed, (int, float))
            
            if first_is_num and not second_is_num:
                value = first_parsed
                col_text = second_group
            else:
                col_text = first_group
                value = second_parsed
            
            col = _match_col(col_text, columns)
            if not col:
                return df, f"I couldn't find a column matching '{col_text.strip()}'. Available columns: {', '.join(columns)}."
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                return df, f"'{col}' isn't numeric, so comparison operators don't apply."
            
            operator = None
            full_match = m.group(0)
            if re.search(r"less\s+than\s+or\s+equal\s+to|<=|<=", full_match):
                operator = "<="
            elif re.search(r"greater\s+than\s+or\s+equal\s+to|>=|>=", full_match):
                operator = ">="
            elif re.search(r"less\s+than|<(?!\s*=|>)", full_match):
                operator = "<"
            elif re.search(r"greater\s+than|>(?!\s*=|>)", full_match):
                operator = ">"
            
            if operator == "<":
                new_df = df[df[col] < value].reset_index(drop=True)
                op_text = "less than"
            elif operator == ">":
                new_df = df[df[col] > value].reset_index(drop=True)
                op_text = "greater than"
            elif operator == "<=":
                new_df = df[df[col] <= value].reset_index(drop=True)
                op_text = "less than or equal to"
            elif operator == ">=":
                new_df = df[df[col] >= value].reset_index(drop=True)
                op_text = "greater than or equal to"
            
            kept = len(new_df)
            if kept == 0:
                return df, f"No rows found where '{col}' is {op_text} {value} — nothing kept (no change made)."
            return new_df, f"Kept {kept:,} row(s) where '{col}' is {op_text} {value}; removed the rest."

    # ---- drop rows where a column starts with / ends with / contains text ----
    # Must run BEFORE the generic "remove exact text from a column" block below,
    # or "drop rows which start with C in InvoiceNo" gets misread as "strip the
    # literal text 'rows which start with c' out of InvoiceNo's values" (a
    # no-op edit, since that phrase never appears in real data) instead of
    # actually dropping the matching rows — which is what was asked for.
    _prefix_specs = [
        ("startswith", r"(?:start(?:s|ing|ed)?|begin(?:s|ning)?)\s+with"),
        ("endswith",   r"end(?:s|ing|ed)?\s+with"),
        ("contains",   r"contains?|includes?"),
    ]
    for str_op, verb_re in _prefix_specs:
        text_val = col_raw = None
        # Form A: "drop rows which start with X in COLUMN"
        m = re.search(rf"\b{REMOVE_VERB_RE}\s+rows?\s+(?:which|that)?\s*(?:{verb_re})\s+(.+?)\s+(?:in|from)\s+(.+)", ql)
        if m:
            text_val, col_raw = m.group(1), m.group(2)
        else:
            # Form B: "drop rows where COLUMN starts with X"
            m = re.search(rf"\b{REMOVE_VERB_RE}\s+rows?\s+where\s+(.+?)\s+(?:{verb_re})\s+(.+)", ql)
            if m:
                col_raw, text_val = m.group(1), m.group(2)
        if m:
            col = _match_col(col_raw, columns)
            text_val = text_val.strip(" '\"")
            if col and text_val:
                series = df[col].astype(str)
                if str_op == "startswith":
                    mask = series.str.lower().str.startswith(text_val.lower())
                    verb_label = "started with"
                elif str_op == "endswith":
                    mask = series.str.lower().str.endswith(text_val.lower())
                    verb_label = "ended with"
                else:
                    mask = series.str.lower().str.contains(re.escape(text_val.lower()), na=False)
                    verb_label = "contained"
                mask = mask & df[col].notna()
                new_df = df[~mask].reset_index(drop=True)
                removed = len(df) - len(new_df)
                if removed == 0:
                    return df, f"No rows found where '{col}' {verb_label} '{text_val}' — nothing removed."
                return new_df, f"Removed {removed:,} row(s) where '{col}' {verb_label} '{text_val}'."

    # ---- remove exact text from a column ----
    m = re.search(rf"\b{REMOVE_VERB_RE}\s+(.+?)\s+(?:in|from)\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        text_to_remove = m.group(1).strip(" '\"")
        col = _match_col(m.group(2), columns)
        
        # Skip if this looks like a comparison operator phrase
        if col and text_to_remove:
            comparison_keywords = ['less than', 'greater than', 'less than or equal', 'greater than or equal', 
                                  'equal to', 'equals', 'is equal', 'is less', 'is greater']
            is_comparison = any(keyword in text_to_remove.lower() for keyword in comparison_keywords)
            
            is_numeric_comparison = bool(re.fullmatch(r'-?\d+\.?\d*\s*(?:<|>|<=|>=|=|==|is|equals?)?\s*-?\d*\.?\d*', 
                                                      text_to_remove, re.IGNORECASE))
            
            if not is_comparison and not is_numeric_comparison:
                pattern = re.escape(text_to_remove)
                if text_to_remove.endswith(".") and len(text_to_remove) > 1:
                    pattern = re.escape(text_to_remove[:-1]) + r"\.?"
                new_df = df.copy()
                # Cast to plain string BEFORE calling .where() — if new_df[col]
                # is still category dtype, .where() tries to insert the
                # post-replace values into that same Categorical, and pandas
                # raises "Cannot setitem on a Categorical with a new category"
                # the moment a replaced value isn't already one of the
                # column's existing categories (which it usually won't be,
                # since we just changed it).
                col_as_str = new_df[col].astype(str)
                cleaned = (
                    col_as_str
                    .where(new_df[col].isna(), col_as_str.str.replace(pattern, "", regex=True, case=False))
                )
                new_df[col] = cleaned.where(cleaned.isna(), cleaned.astype(str).str.replace(r"\s+", " ", regex=True).str.strip())
                changed = int((df[col].astype(str) != new_df[col].astype(str)).sum())
                if changed == 0:
                    return df, f"No '{text_to_remove}' text found in '{col}' — nothing removed."
                return new_df, f"Removed '{text_to_remove}' from {changed:,} value(s) in '{col}'."

    # ---- remove outliers (drop the rows, not cap them) ----
    m = re.search(rf"\b(?:{REMOVE_VERB_RE}|caps?)\s+outliers?\s*(?:in\s+|from\s+)?(.+)", ql) or \
        re.search(r"outliers?\s+(?:in\s+|from\s+)?(.+)", ql)
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

    # ---- replace value in column ----
    _rep_m = (
        re.search(r"\breplace\s+(.+?)\s+with\s*(.+?)\s+in\s+(.+)", q, re.IGNORECASE)
        or re.search(r"in\s+(.+?)\s+replace\s+(.+?)\s+with\s*(.+)", q, re.IGNORECASE)
        or re.search(r"replace\s+(?:the\s+)?(.+?)\s+(?:to|into)\s*(.+?)\s+(?:in\s+)?(.+)", q, re.IGNORECASE)
        or re.search(r"change\s+(.+?)\s+(?:to|into)\s+(.+?)\s+in\s+(.+)", q, re.IGNORECASE)
        or re.search(r"swap\s+(.+?)\s+(?:and|for)\s+(.+?)\s+in\s+(.+)", q, re.IGNORECASE)
        or re.search(r"(?:set|make)\s+(.+?)\s+(?:to|as|equal\s+to)\s+(.+?)\s+in\s+(.+)", q, re.IGNORECASE)
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
        if pd.api.types.is_numeric_dtype(col_series):
            old_num = _parse_value_token(old_val_raw)
            new_num = _parse_value_token(new_val_raw)
            if old_num is not None and new_num is not None:
                mask = col_series == old_num
                count = int(mask.sum())
                if count == 0:
                    return df, f"No rows in '{col}' equal {old_val_raw} — nothing changed."
                new_df[col] = col_series.where(~mask, new_num)
                return new_df, f"Replaced {count:,} occurrence(s) of {old_val_raw} → {new_val_raw} in '{col}'."
        mask = col_series.astype(str).str.strip().str.lower() == old_val_raw.lower()
        count = int(mask.sum())
        if count == 0:
            mask = col_series.astype(str).str.lower().str.contains(re.escape(old_val_raw.lower()), na=False)
            count = int(mask.sum())
            if count == 0:
                return df, f"No rows in '{col}' contain '{old_val_raw}' — nothing changed."
            new_df[col] = col_series.where(
                ~mask,
                col_series.astype(str).str.replace(old_val_raw, new_val_raw, case=False, regex=False)
            )
            return new_df, f"Replaced '{old_val_raw}' → '{new_val_raw}' in {count:,} cell(s) of '{col}' (substring match)."
        new_df[col] = col_series.where(~mask, new_val_raw)
        return new_df, f"Replaced {count:,} occurrence(s) of '{old_val_raw}' → '{new_val_raw}' in '{col}'."

    # ---- filter rows by value (remove / keep only) ----
    # NOT_EQ_OP must be tried BEFORE EQ_OP: "is not equal to" contains "is" and
    # "equal to", so if EQ_OP were checked first it would wrongly grab "is" as
    # the separator and leave "not equal to 0" dangling as the literal value.
    NOT_EQ_OP = r"(?:is\s+not\s+equal\s+to|is\s+not|not\s+equal\s+to|!=|<>)"
    EQ_OP = r"(?:is\s+equal\s+to|equal\s+to|equals\s+to|==|=|equals?|is)"

    m = re.search(rf"\b{REMOVE_VERB_RE}\s+rows?\s+where\s+(.+?)\s+{NOT_EQ_OP}\s+(.+)", ql) or \
        re.search(rf"delete\s+(?:rows?\s+)?where\s+(.+?)\s+{NOT_EQ_OP}\s+(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        value = _parse_value_token(m.group(2))
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        mask = df[col].astype(str).str.strip().str.lower() == str(value).strip().lower()
        removed = int((~mask).sum())
        if removed == 0:
            return df, f"No rows found where '{col}' is not '{value}' — nothing removed."
        return df[mask].reset_index(drop=True), f"Removed {removed:,} row(s) where '{col}' was not '{value}'."

    m = re.search(rf"\b{REMOVE_VERB_RE}\s+rows?\s+where\s+(.+?)\s+{EQ_OP}\s+(.+)", ql) or \
        re.search(rf"delete\s+(?:rows?\s+)?where\s+(.+?)\s+{EQ_OP}\s+(.+)", ql) or \
        re.search(rf"{REMOVE_VERB_RE}\s+all\s+(?:the\s+)?(?:rows?\s+)?where\s+(.+?)\s+{EQ_OP}\s+(.+)", ql)
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

    m = re.search(rf"\bkeep\s+only\s+rows?\s+where\s+(.+?)\s+{NOT_EQ_OP}\s+(.+)", ql) or \
        re.search(rf"filter\s+(?:only\s+)?(?:rows?\s+)?where\s+(.+?)\s+{NOT_EQ_OP}\s+(.+)", ql) or \
        re.search(rf"show\s+(?:only\s+)?(?:rows?\s+)?where\s+(.+?)\s+{NOT_EQ_OP}\s+(.+)", ql)
    if m:
        col = _match_col(m.group(1), columns)
        value = _parse_value_token(m.group(2))
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        mask = df[col].astype(str).str.strip().str.lower() != str(value).strip().lower()
        kept = int(mask.sum())
        if kept == 0:
            return df, f"No rows found where '{col}' is not '{value}' — nothing kept (no change made)."
        return df[mask].reset_index(drop=True), f"Kept {kept:,} row(s) where '{col}' is not '{value}'; removed the rest."

    m = re.search(rf"\bkeep\s+only\s+rows?\s+where\s+(.+?)\s+{EQ_OP}\s+(.+)", ql) or \
        re.search(rf"filter\s+(?:only\s+)?(?:rows?\s+)?where\s+(.+?)\s+{EQ_OP}\s+(.+)", ql) or \
        re.search(rf"show\s+(?:only\s+)?(?:rows?\s+)?where\s+(.+?)\s+{EQ_OP}\s+(.+)", ql)
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

    return df, (
        "I didn't recognise that command. " + HELP_TEXT
    )