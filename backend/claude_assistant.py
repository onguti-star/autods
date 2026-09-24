"""
Online AI assistant (Claude) -- an OPTIONAL, additional helper that sits
alongside AutoDS's existing offline assistant (backend/assistant.py, used by
/api/ask). This one calls the real Anthropic API, so it needs an internet
connection and the person's own API key; the offline assistant keeps working
exactly as before with neither.

Design choices worth knowing:
- The API key never leaves the backend. It's saved to a file in the app's
  data directory and only ever reported back to the frontend as a boolean.
- Claude is never shown the raw dataset -- only the same kind of profile
  summary already used elsewhere in the app (shape, dtypes, missingness,
  stats, top categories, correlations). No row-level values are sent unless
  they already appear in that summary (small samples of category names).
- Claude never edits the data directly. When it wants to suggest an action,
  it calls the `suggest_action` tool, and the frontend shows that as a
  clickable chip. Clicking a "run_command" chip sends the exact same command
  text to the EXISTING /api/ask/{session_id} endpoint -- the same rule-based
  executor the offline assistant already uses, with the same safety and undo
  support. Claude only ever proposes text; it never touches the DataFrame.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import eda, narrate, store

router = APIRouter(prefix="/api/claude", tags=["claude"])

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = os.environ.get("AUTODS_CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOKENS = 1024
REQUEST_TIMEOUT = 60

# Same data directory desktop mode already uses for sessions; falls back to
# a project-local folder for the plain "python backend/run.py" case.
_DATA_DIR = os.environ.get("AUTODS_DATA_DIR", "").strip()
if _DATA_DIR:
    _CONFIG_DIR = os.path.abspath(_DATA_DIR)
else:
    _CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".sessions"))
os.makedirs(_CONFIG_DIR, exist_ok=True)
_KEY_PATH = os.path.join(_CONFIG_DIR, "claude_api_key.txt")

SYSTEM_PROMPT = """You are an embedded data-science assistant inside AutoDS, a \
desktop app for cleaning, visualising and training models on tabular data. \
You are talking with someone building a data science portfolio; be concrete \
and practical, not generic.

You do NOT see the raw data, only a text profile of it (shape, column types, \
missing %, basic stats, top category values, correlations). Answer from that; \
don't invent exact values it doesn't contain.

When you want to propose something the person can actually do in the app, \
call the suggest_action tool (you may call it more than once per turn). Use \
run_command for anything the app's own cleaning engine can do -- it understands \
short natural-language instructions such as:
  "remove duplicate rows"
  "remove rows where <column> is <value>"
  "fill missing <column> with <value>" / "fill missing <column> with the mean"
  "drop column <column>"
  "rename <column> to <new name>"
  "set <column> to <value> where <column> is <value>"
Phrase the command field exactly the way a person would type it -- it is sent \
verbatim to that engine, not interpreted by you. Use open_tab to point at a \
different part of the app (Ingest, Profile, Clean, Visualise, Training, Models). \
Use build_chart for a specific chart worth looking at. Keep each suggestion's \
reason to one short sentence. Don't call the tool for vague ideas you're not \
confident the command engine can actually run -- suggest those in plain text \
instead."""

SUGGEST_ACTION_TOOL = {
    "name": "suggest_action",
    "description": (
        "Propose one concrete, clickable next step for the person to take in AutoDS. "
        "Call this once per suggestion; call it several times for several suggestions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["run_command", "open_tab", "build_chart"],
                "description": "run_command: a cleaning instruction for AutoDS's own command engine. "
                                "open_tab: send the person to a specific tab. "
                                "build_chart: suggest a specific chart to build in Visualise.",
            },
            "command": {
                "type": "string",
                "description": "Required for run_command. A short natural-language instruction, "
                                "e.g. \"remove rows where age is null\".",
            },
            "tab": {
                "type": "string",
                "enum": ["Ingest", "Profile", "Clean", "Visualise", "Training", "Models"],
                "description": "Required for open_tab.",
            },
            "chart": {
                "type": "object",
                "description": "Required for build_chart.",
                "properties": {
                    "chart_type": {"type": "string", "enum": [
                        "histogram", "bar", "pie", "boxplot", "scatter", "line", "bubble", "area",
                        "heatmap", "treemap", "radar", "violin", "density", "choropleth",
                    ]},
                    "x": {"type": "string", "description": "Column name for the x axis / category."},
                    "y": {"type": "string", "description": "Column name for the y axis, if the chart needs one."},
                    "group": {"type": "string", "description": "Optional grouping column."},
                },
                "required": ["chart_type", "x"],
            },
            "reason": {"type": "string", "description": "One short sentence: why this is worth doing."},
        },
        "required": ["type", "reason"],
    },
}


# --------------------------------------------------------------------- key
def _read_key() -> str:
    try:
        with open(_KEY_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _write_key(key: str) -> None:
    fd = os.open(_KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key.strip())


class KeyIn(BaseModel):
    api_key: str


@router.get("/status")
def status():
    key = _read_key()
    masked = f"sk-...{key[-4:]}" if len(key) >= 8 else None
    return {"configured": bool(key), "model": DEFAULT_MODEL, "masked_key": masked}


@router.post("/key")
def set_key(body: KeyIn):
    key = body.api_key.strip()
    if not key:
        raise HTTPException(400, "API key is empty.")
    if len(key) < 12:
        raise HTTPException(400, "That doesn't look like a full API key.")
    _write_key(key)
    return status()


@router.delete("/key")
def clear_key():
    try:
        os.remove(_KEY_PATH)
    except FileNotFoundError:
        pass
    return status()


# --------------------------------------------------------------- summary
def _fmt_num(v: Any) -> str:
    if v is None:
        return "?"
    if isinstance(v, float):
        return f"{v:,.3g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _build_summary_text(df) -> str:
    profile = eda.profile_dataframe(df)
    lines = []
    shape = profile["shape"]
    lines.append(f"Dataset: {shape['rows']:,} rows x {shape['columns']} columns.")
    if profile.get("duplicate_rows"):
        lines.append(f"{profile['duplicate_rows']:,} duplicate row(s).")
    total_cells = profile.get("total_cells") or (shape["rows"] * shape["columns"]) or 1
    miss_pct = 100 * profile.get("total_missing_cells", 0) / total_cells
    lines.append(f"Missing data: {profile.get('total_missing_cells', 0):,} cells ({miss_pct:.1f}% overall).")
    lines.append("")
    lines.append("Columns:")
    for c in profile.get("columns", []):
        bits = [f"- {c['name']} ({c.get('dtype', c.get('type', '?'))})"]
        miss = c.get("missing_count") or c.get("missing", 0)
        if miss:
            pct = c.get("missing_pct")
            bits.append(f"{miss:,} missing" + (f" ({pct:.1f}%)" if pct is not None else ""))
        if c.get("type") == "numeric" and c.get("stats"):
            s = c["stats"]
            bits.append(f"mean {_fmt_num(s.get('mean'))}, min {_fmt_num(s.get('min'))}, max {_fmt_num(s.get('max'))}")
        elif c.get("type") in ("categorical", "text") and c.get("top_values"):
            top = ", ".join(f"{t.get('value')} ({t.get('count')})" for t in c["top_values"][:5])
            bits.append(f"top values: {top}")
        elif c.get("unique") is not None:
            bits.append(f"{c['unique']:,} unique values")
        lines.append("; ".join(bits))
    rec = profile.get("learning_recommendation")
    if isinstance(rec, dict):
        lines.append("")
        if rec.get("summary"):
            lines.append(f"Suggested approach: {rec['summary']}")
        if rec.get("recommended_next_step"):
            lines.append(f"Recommended next step: {rec['recommended_next_step']}")
        targets = rec.get("suggested_targets") or []
        if targets:
            picks = ", ".join(f"{t['column']} ({t.get('problem_type', '?')})" for t in targets[:3])
            lines.append(f"Good target column candidates: {picks}")
    elif rec:
        lines.append("")
        lines.append(f"Note: {rec}")
    try:
        narrative = narrate.narrate_eda(profile)
        if narrative:
            lines.append("")
            lines.append("Automated notes:")
            lines.extend(f"- {n}" for n in narrative[:8])
    except Exception:
        pass
    return "\n".join(lines)


@router.get("/summary/{session_id}")
def get_summary(session_id: str):
    try:
        session = store.get_session(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found. Load a dataset first.")
    return {"summary": _build_summary_text(session.df)}


# ------------------------------------------------------------------ chat
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[ChatMessage]


def _call_anthropic(api_key: str, messages: list[dict]) -> dict:
    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": [SUGGEST_ACTION_TOOL],
    }
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error", {}).get("message", body)
        except Exception:
            detail = body
        if e.code == 401:
            raise HTTPException(401, "Claude rejected that API key. Check it in settings.")
        if e.code == 429:
            raise HTTPException(429, "Rate-limited by the Claude API. Try again shortly.")
        raise HTTPException(502, f"Claude API error: {detail[:300]}")
    except urllib.error.URLError as e:
        raise HTTPException(503, f"Couldn't reach the Claude API (no internet?): {e.reason}")


@router.post("/chat/{session_id}")
def chat(session_id: str, body: ChatIn):
    api_key = _read_key()
    if not api_key:
        raise HTTPException(400, "No Claude API key saved yet. Add one in the assistant's settings.")
    try:
        store.get_session(session_id)  # just confirms the dataset session is real
    except KeyError:
        raise HTTPException(404, "Session not found. Load a dataset first.")
    if not body.messages:
        raise HTTPException(400, "No message to send.")

    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    data = _call_anthropic(api_key, messages)

    reply_parts, actions = [], []
    for block in data.get("content", []):
        if block.get("type") == "text":
            reply_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use" and block.get("name") == "suggest_action":
            actions.append(block.get("input", {}))
    return {"reply": "\n".join(reply_parts).strip(), "actions": actions}
