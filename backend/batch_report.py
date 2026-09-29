"""Text for batch predictions in downloaded reports and notebooks.

A batch prediction is stored as a record in two sessions (see predict_batch in
main.py):

* the MODEL's dataset (e.g. train.csv)   -> role is None      ("N rows scored")
* the SCORED dataset  (e.g. test.csv)    -> role == "scored"  ("column added here")

Both the HTML report, the Markdown work report and the .ipynb export use these
helpers, so a batch prediction reads the same everywhere instead of being shown
as if it were one hand-typed row.
"""
from __future__ import annotations

import html as _html
from typing import Any

PREVIEW_ROWS = 10
MAX_MARKDOWN_COLUMNS = 8


def is_batch(prediction: dict) -> bool:
    return bool(prediction.get("is_batch"))


def is_scored_record(prediction: dict) -> bool:
    return prediction.get("role") == "scored"


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        if value != value:          # NaN
            return "missing"
        return f"{value:,.4g}"
    return str(value).replace("\n", " ").replace("\r", " ")


def headline(prediction: dict) -> str:
    target = prediction.get("target") or "prediction"
    rows = int(prediction.get("rows_predicted") or 0)
    if is_scored_record(prediction):
        return f"Batch predictions added to this dataset: {target} ({rows:,} rows)"
    return f"Batch prediction: {target} ({rows:,} rows scored)"


def detail_lines(prediction: dict) -> list[str]:
    """Plain-text facts, one per line (no markup)."""
    source = prediction.get("source_name") or "Current training"
    model = prediction.get("model_name") or "model"
    column = prediction.get("saved_column") or "prediction"
    scored = prediction.get("scored_dataset") or "the scored dataset"
    trained_on = prediction.get("model_dataset")

    lines = []
    if is_scored_record(prediction):
        lines.append(f"Prediction column on this dataset: {column}")
        lines.append(f"Model: {source} ({model})" + (f", trained on '{trained_on}'" if trained_on else ""))
    else:
        lines.append(f"Scored dataset: '{scored}' (predictions saved there as column '{column}')")
        lines.append(f"Model: {source} ({model})")
    if prediction.get("id_column"):
        lines.append(f"Row id column: {prediction['id_column']}")
    missing = prediction.get("missing_feature_columns") or []
    if missing:
        lines.append("Feature columns missing from the scored dataset (filled blank before scoring): "
                     + ", ".join(str(c) for c in missing))
    return lines


def preview_columns_rows(prediction: dict, max_columns: int | None = None):
    """(columns, rows-of-strings) for the stored preview, or ([], []) if none."""
    rows = prediction.get("preview") or []
    if not rows:
        return [], []
    columns = list(rows[0].keys())
    if max_columns and len(columns) > max_columns:
        target = prediction.get("target")
        keep = columns[:max_columns]
        if target in columns and target not in keep:
            keep[-1] = target
        columns = keep
    return columns, [[_fmt(r.get(c)) for c in columns] for r in rows[:PREVIEW_ROWS]]


# ------------------------------------------------------------------ HTML
def html_block(prediction: dict) -> str:
    esc = lambda v: _html.escape(str(v))  # noqa: E731
    meta = " · ".join(str(x) for x in (
        prediction.get("source_name") or "Current training",
        prediction.get("model_name") or "model",
        prediction.get("created_at") or "",
    ) if x)
    parts = [
        '        <div class="prediction-item">',
        f'            <div class="prediction-output">{esc(headline(prediction))}</div>',
        f'            <div class="prediction-meta">{esc(meta)}</div>',
    ]
    for line in detail_lines(prediction):
        parts.append(f'            <div class="prediction-inputs">{esc(line)}</div>')

    columns, rows = preview_columns_rows(prediction)
    if columns:
        parts.append(f'            <div class="prediction-inputs"><strong>First {len(rows)} rows scored:</strong></div>')
        parts.append('            <div class="batch-preview-wrap"><table class="batch-preview"><thead><tr>'
                     + "".join(f"<th>{esc(c)}</th>" for c in columns)
                     + "</tr></thead><tbody>"
                     + "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
                     + "</tbody></table></div>")
    parts.append("        </div>")
    return "\n".join(parts)


BATCH_CSS = """
        .batch-preview-wrap {
            overflow-x: auto;
            margin-top: 8px;
            max-width: 100%;
        }
        .batch-preview {
            border-collapse: collapse;
            font-size: 0.85em;
            background: #fff;
        }
        .batch-preview th,
        .batch-preview td {
            border: 1px solid #e3e6ea;
            padding: 4px 9px;
            text-align: left;
            white-space: nowrap;
        }
        .batch-preview th {
            background: #f3f5f7;
        }
"""


# -------------------------------------------------------------- Markdown
def markdown_lines(prediction: dict) -> list[str]:
    lines = [f"- **{headline(prediction)}**"]
    for line in detail_lines(prediction):
        lines.append(f"  - {line}")
    if prediction.get("created_at"):
        lines.append(f"  - Saved: {prediction['created_at']}")
    columns, rows = preview_columns_rows(prediction, MAX_MARKDOWN_COLUMNS)
    if columns:
        lines.extend(["", f"  First {len(rows)} rows scored:", ""])
        lines.append("  | " + " | ".join(columns) + " |")
        lines.append("  | " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("  | " + " | ".join(v.replace("|", "\\|") for v in row) + " |")
        lines.append("")
    return lines


# -------------------------------------------------------------- Notebook
def notebook_markdown(prediction: dict) -> str:
    text = f"### {headline(prediction)}\n\n"
    text += "\n".join(f"- {line}" for line in detail_lines(prediction))
    if prediction.get("created_at"):
        text += f"\n- Saved: {prediction['created_at']}"
    if is_scored_record(prediction):
        text += "\n\nThe prediction column is already part of the data loaded at the top of this notebook."
    else:
        text += "\n\nThe full scored file can be downloaded from the model card in AutoDS; the first rows are shown below."
    return text
