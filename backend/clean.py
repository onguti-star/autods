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
    "• drop column notes\n"
    "• rename column dob to date_of_birth\n"
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
    "• convert price to number  (or: integer / text / category / date / time / boolean)\n"
    "• change ORDER_DATE to date / time\n"
    "• reset  (undoes every chat cleaning command and restores the original upload)\n"
    "\n"
    "📂 Split into a new tab:\n"
    "• split rows where gender is Male into new tab\n"
    "• send rows where age is 3 to new tab\n"
    "• create new tab where country is Kenya\n"
    "• new tab where status is cancelled  (keeps original intact)\n"
    "\n"
    "Math operations: +, -, *, /, ** (power), % (modulo)\n"
    "Examples: 'price * 1.2', 'quantity + 10', '(price - cost) / cost * 100'"
)


def run_command(df: pd.DataFrame, text: str, original_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str | dict]:
    corrected_text, corrections = _correct_keyword_typos(text.strip(), CLEAN_KEYWORDS)
    new_df, message = _run_command_impl(df, corrected_text, original_df)
    # split_to_tab returns a dict — don't try to prepend a string correction note to it
    if isinstance(message, dict):
        return new_df, message
    return new_df, _format_keyword_correction_note(corrections) + message


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

    # ---- split / send rows to a new dataset tab (supports multiple conditions) ----
    # Accepts many phrasings, word-order variations, and common misspellings.
    # Supports: "split where gender is Male and age is 3 into new tab"
    #           "new tab where country is Kenya or country is Uganda"
    _SPLIT_VERBS  = r"(?:split|spilt|splitt?|send|sned|snde|export|extract|copy|move|push|filter|separate)"
    _CREATE_VERBS = r"(?:create|creat|craete|make|open|add)"
    _NEW_WORD     = r"(?:new|nwe|enw)"
    _TAB_WORD     = r"(?:tab|tba|dataset)"
    _WHERE_WORD   = r"(?:where|when|with|for|that\s+(?:have|has|are|is))"
    _IS_WORD      = r"(?:is|=|==|equals?|are)"

    _tab_trigger = re.search(
        rf"""
        (?:
            {_SPLIT_VERBS}
            \s+(?:rows?\s+)?(?:{_WHERE_WORD})\s+
            (.+?)
            \s+(?:into?|to|as)\s+(?:a\s+)?(?:{_NEW_WORD}\s+)?{_TAB_WORD}
            |
            {_CREATE_VERBS}\s+(?:a\s+)?(?:{_NEW_WORD}\s+)?{_TAB_WORD}\s+
            (?:{_WHERE_WORD})\s+(.+)
            |
            (?:{_NEW_WORD}\s+{_TAB_WORD})\s+(?:{_WHERE_WORD})\s+(.+)
        )
        """,
        ql,
        re.VERBOSE,
    )

    if _tab_trigger:
        # Pull the conditions string from whichever branch matched
        conditions_raw = next((g for g in _tab_trigger.groups() if g), None)

        if conditions_raw:
            # ---- parse multi-condition string ----
            # Split on " and " / " or " keeping track of the logic operator
            # e.g. "gender is Male and age is 3 or country is Kenya"
            #   → [("and", "gender is Male"), ("or", "age is 3"), ("or", "country is Kenya")]
            def _parse_conditions(cond_str):
                """
                Returns list of (logic, col, op, value) tuples.
                logic is 'and'|'or' (first entry is always 'and').
                op is '=='|'!='|'>'|'>='|'<'|'<='|'contains'.
                """
                # Tokenise on ' and ' / ' or '
                parts = re.split(r'\s+(and|or)\s+', cond_str.strip(), flags=re.IGNORECASE)
                # parts alternates: [cond, logic, cond, logic, cond ...]
                result = []
                logic = "and"
                for part in parts:
                    p = part.strip()
                    if p.lower() in ("and", "or"):
                        logic = p.lower()
                        continue
                    # Match: col [is|>|<|>=|<=|!=|contains] value
                    m = re.match(
                        rf"(.+?)\s+(?:{_IS_WORD}|contains?|greater\s+than|less\s+than|not\s+equal|!=|>=|<=|>(?!=)|<(?!=))\s+(.+)",
                        p, re.IGNORECASE
                    )
                    if not m:
                        result.append((logic, None, None, None, p))   # unparseable
                        logic = "and"
                        continue

                    raw_col = m.group(1).strip()
                    raw_val = m.group(2).strip(" '\"?.,:;-")

                    # Detect operator from the full part
                    if re.search(r'\bgreater\s+than\s+or\s+equal|>=', p, re.I): op = ">="
                    elif re.search(r'\bless\s+than\s+or\s+equal|<=', p, re.I): op = "<="
                    elif re.search(r'\bgreater\s+than|>(?!=)', p, re.I):        op = ">"
                    elif re.search(r'\bless\s+than|<(?!=)', p, re.I):           op = "<"
                    elif re.search(r'\bnot\s+equal|!=', p, re.I):               op = "!="
                    elif re.search(r'\bcontains?', p, re.I):                    op = "contains"
                    else:                                                        op = "=="

                    result.append((logic, raw_col, op, raw_val, p))
                    logic = "and"
                return result

            parsed = _parse_conditions(conditions_raw)

            # Validate columns
            errors = []
            resolved = []
            for logic, raw_col, op, raw_val, original in parsed:
                if raw_col is None:
                    errors.append(f"Couldn't parse condition: '{original}'")
                    continue
                col = _match_col(raw_col, columns)
                if not col:
                    errors.append(f"Column '{raw_col.strip()}' not found. Available: {', '.join(columns)}")
                    continue
                value = _parse_value_token(raw_val)
                resolved.append((logic, col, op, value))

            if errors:
                return df, (
                    "Some conditions couldn't be understood:\n" +
                    "\n".join(f"• {e}" for e in errors) + "\n\n"
                    "Example: split rows where gender is Male and age is 3 into new tab"
                )

            # Build the combined mask
            mask = pd.Series([True] * len(df), index=df.index)
            condition_parts = []
            for i, (logic, col, op, value) in enumerate(resolved):
                num_val = value if isinstance(value, (int, float)) else None
                if op == "==" :
                    col_mask = df[col].astype(str).str.strip().str.lower() == str(value).lower()
                elif op == "!=":
                    col_mask = df[col].astype(str).str.strip().str.lower() != str(value).lower()
                elif op == ">" and num_val is not None:
                    col_mask = pd.to_numeric(df[col], errors="coerce") > num_val
                elif op == ">=" and num_val is not None:
                    col_mask = pd.to_numeric(df[col], errors="coerce") >= num_val
                elif op == "<" and num_val is not None:
                    col_mask = pd.to_numeric(df[col], errors="coerce") < num_val
                elif op == "<=" and num_val is not None:
                    col_mask = pd.to_numeric(df[col], errors="coerce") <= num_val
                elif op == "contains":
                    col_mask = df[col].astype(str).str.contains(str(value), case=False, na=False)
                else:
                    col_mask = df[col].astype(str).str.strip().str.lower() == str(value).lower()

                if i == 0:
                    mask = col_mask
                elif logic == "or":
                    mask = mask | col_mask
                else:
                    mask = mask & col_mask

                op_label = {"==": "is", "!=": "is not", ">": ">", ">=": ">=", "<": "<", "<=": "<=", "contains": "contains"}.get(op, op)
                prefix = "" if i == 0 else logic.upper() + " "
                condition_parts.append(f"{prefix}{col} {op_label} {value}")

            matched = int(mask.sum())
            conditions_label = " | ".join(condition_parts)

            if matched == 0:
                return df, (
                    f"No rows matched: {conditions_label}\n"
                    "Try loosening one of the conditions."
                )

            subset = df[mask].reset_index(drop=True)
            tab_name = " & ".join(
                f"{col}={val}" for _, col, _, val in resolved
            )[:60]   # cap length

            return df, {
                "__action__": "split_to_tab",
                "conditions_label": conditions_label,
                "rows_matched": matched,
                "total_rows": len(df),
                "subset_df": subset,          # full DataFrame — used by endpoint
                "subset_records": subset.head(5).to_dict(orient="records"),
                "columns": list(df.columns),
                "tab_name": tab_name,
            }


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
        empty_col_name = _parse_empty_column_name(q)
        if empty_col_name:
            if empty_col_name in columns:
                return df, f"Column '{empty_col_name}' already exists."
            new_df = df.copy()
            new_df[empty_col_name] = pd.NA
            return new_df, f"Created empty column '{empty_col_name}'."

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
            
            return new_df, f"Created new column '{new_col_name}' as {op_name} of all numeric columns"
        
        # Otherwise, try to parse as a regular expression
        expr, new_col_name = _parse_math_expression(q, df)
        
        if expr and new_col_name:
            if new_col_name in columns:
                return df, f"Column '{new_col_name}' already exists. Use: fill {new_col_name} with {expr}"
            return _apply_math_expression_to_column(df, new_col_name, expr, "Created new column")
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
    m = re.search(r"\b(?:drop|remove|delete|discard|get\s+rid\s+of)\s+(?:the\s+)?column\s+(.+)", ql) or \
        re.search(r"\b(?:drop|remove|delete|discard)\s+(.+?)\s+column\b", ql) or \
        re.search(r"column\s+(.+?)\s+(?:drop|remove|delete|discard)\b", ql)
    if m:
        col = _match_col(m.group(1), columns) or _match_col(q, columns)
        if not col:
            return df, f"I couldn't find a column matching '{m.group(1).strip()}'. Available columns: {', '.join(columns)}."
        return df.drop(columns=[col]), f"Dropped column '{col}'."

    # ---- rename column ----
    m = re.search(r"\brename\s+(?:the\s+)?(?:column\s+)?(.+?)\s+(?:to|into|as)\s+(.+)", ql) or \
        re.search(r"change\s+(?:the\s+)?(?:name\s+of\s+)?(?:column\s+)?(.+?)\s+(?:to|into|as)\s+(.+)", ql)
    if m:
        old_col = _match_col(m.group(1), columns)
        new_name = m.group(2).strip(" '\"?.,:;-")
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
    m = re.search(r"\b(?:remove|drop|delete)\s+rows?\s+where\s+(.+?)\s+is\s+(?:missing|null|na|empty)\b", ql) or \
        re.search(r"\b(?:remove|drop|delete)\s+(?:rows?\s+with\s+)?(?:missing|null|empty)\s+(?:values?\s+in\s+)?(.+)", ql) or \
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
        r"\b(?:remove|drop|delete)\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:less\s+than\s+or\s+equal\s+to|<=)\s+(.+)",
        r"\b(?:remove|drop|delete)\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:greater\s+than\s+or\s+equal\s+to|>=)\s+(.+)",
        r"\b(?:remove|drop|delete)\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:less\s+than|<)\s+(.+)",
        r"\b(?:remove|drop|delete)\s+rows?\s+where\s+(.+?)\s+(?:is\s+)?(?:greater\s+than|>)\s+(.+)",
        # Pattern: "remove rows in col where values is less than value"
        r"\b(?:remove|drop|delete)\s+rows?\s+(?:in|from)\s+(.+?)\s+where\s+(?:values?\s+)?(?:is\s+)?(?:less\s+than\s+or\s+equal\s+to|<=)\s+(.+)",
        r"\b(?:remove|drop|delete)\s+rows?\s+(?:in|from)\s+(.+?)\s+where\s+(?:values?\s+)?(?:is\s+)?(?:greater\s+than\s+or\s+equal\s+to|>=)\s+(.+)",
        r"\b(?:remove|drop|delete)\s+rows?\s+(?:in|from)\s+(.+?)\s+where\s+(?:values?\s+)?(?:is\s+)?(?:less\s+than|<)\s+(.+)",
        r"\b(?:remove|drop|delete)\s+rows?\s+(?:in|from)\s+(.+?)\s+where\s+(?:values?\s+)?(?:is\s+)?(?:greater\s+than|>)\s+(.+)",
        # Pattern: "remove less than 0 in quantity" or "remove greater than 100 from price"
        r"\b(?:remove|drop|delete)\s+(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:less\s+than\s+or\s+equal\s+to|<=)\s+(.+?)\s+(?:in|from)\s+(.+)",
        r"\b(?:remove|drop|delete)\s+(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:greater\s+than\s+or\s+equal\s+to|>=)\s+(.+?)\s+(?:in|from)\s+(.+)",
        r"\b(?:remove|drop|delete)\s+(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:less\s+than|<)\s+(.+?)\s+(?:in|from)\s+(.+)",
        r"\b(?:remove|drop|delete)\s+(?:rows?\s+)?(?:where\s+)?(?:values?\s+)?(?:greater\s+than|>)\s+(.+?)\s+(?:in|from)\s+(.+)",
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

    # ---- remove exact text from a column ----
    m = re.search(r"\bremove\s+(.+?)\s+(?:in|from)\s+(.+)", q, flags=re.IGNORECASE)
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
    m = re.search(r"\b(?:remove|caps?|delete|drop)\s+outliers?\s*(?:in\s+|from\s+)?(.+)", ql) or \
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
    m = re.search(r"\b(?:remove|drop|delete)\s+rows?\s+where\s+(.+?)\s+(?:is|=|==|equals?)\s+(.+)", ql) or \
        re.search(r"delete\s+(?:rows?\s+)?where\s+(.+?)\s+(?:is|=|==|equals?)\s+(.+)", ql) or \
        re.search(r"(?:delete|remove)\s+all\s+(?:the\s+)?(?:rows?\s+)?where\s+(.+?)\s+(?:is|=|==)\s+(.+)", ql)
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

    m = re.search(r"\bkeep\s+only\s+rows?\s+where\s+(.+?)\s+(?:is|=|==|equals?)\s+(.+)", ql) or \
        re.search(r"filter\s+(?:only\s+)?(?:rows?\s+)?where\s+(.+?)\s+(?:is|=|==)\s+(.+)", ql) or \
        re.search(r"show\s+(?:only\s+)?(?:rows?\s+)?where\s+(.+?)\s+(?:is|=|==)\s+(.+)", ql)
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