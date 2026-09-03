import asyncio
import io
import html
import ipaddress
import logging
import math
import multiprocessing as mp
import os
import pickle
import socket
import sqlite3
import tempfile
import urllib.parse
import urllib.request
import json
import uuid
from datetime import datetime
from typing import Literal

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import assistant
from . import automl
from . import clean as clean_module
from . import clean_chat
from . import eda
from . import geo
from . import narrate
from . import nlp
from . import nb as nb_module
from . import pca_analysis
from . import simulation as simulation_module
from . import unsupervised
from . import viz
from .main_report import _build_html_report
from .store import (
    SESSIONS,
    create_session,
    get_session,
    delete_session,
    purge_stale_session_files,
    evict_idle_sessions,
)

logger = logging.getLogger("autods.main")

app = FastAPI(title="AutoDS")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_REMOTE_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_DATAFRAME_ROWS = 5_500_000  # 5.5 million rows
MAX_DATAFRAME_COLUMNS = 1_000  # 1000 columns

# CORS: open by default for local dev (frontend runs on a different port than
# the API), but scopable via env var for real deployments — set
# ALLOWED_ORIGINS to a comma-separated list (e.g. "https://myapp.netlify.app")
# so a browser page on some other origin can't call this API using a user's
# cookies/credentials. Unset/"*" keeps today's wide-open local-dev behavior.
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = (
    ["*"] if _allowed_origins_env == "*"
    else [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_cleanup_task: "asyncio.Task | None" = None
_SESSION_CLEANUP_INTERVAL_SECONDS = 60 * 60  # sweep hourly


async def _periodic_session_cleanup(interval_seconds: int = _SESSION_CLEANUP_INTERVAL_SECONDS):
    """Runs for the life of the server: evicts in-memory sessions nobody has
    touched in 24h (freeing RAM) and removes orphaned .sessions/ disk files
    older than 7 days (freeing disk). Previously this only ran once at
    startup, so a long-lived server process would grow both memory and disk
    usage without bound between restarts."""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            evicted = evict_idle_sessions(max_idle_hours=24)
            removed = purge_stale_session_files(max_age_days=7)
            if evicted or removed:
                logger.info(
                    "session cleanup: evicted %d idle sessions from memory, removed %d stale files from disk",
                    evicted, removed,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("periodic session cleanup failed")


@app.on_event("startup")
def _cleanup_stale_sessions_on_startup():
    """Sweep .sessions/ for CSV files from long-closed or crashed sessions so
    disk usage doesn't grow unbounded run after run, then kick off a
    recurring background sweep so the same cleanup keeps happening while the
    server stays up (not just once at boot)."""
    purge_stale_session_files(max_age_days=7)
    global _session_cleanup_task
    _session_cleanup_task = asyncio.create_task(_periodic_session_cleanup())


@app.on_event("shutdown")
async def _stop_session_cleanup():
    if _session_cleanup_task is not None:
        _session_cleanup_task.cancel()



@app.exception_handler(500)
async def internal_exception_handler(request, exc):
    """Ensure all 500 errors return JSON, not HTML."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all handler to ensure all exceptions return JSON."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )


# ---------- Upload ----------

def _validate_dataframe_size(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    if rows > MAX_DATAFRAME_ROWS:
        raise HTTPException(
            400,
            f"Dataset has {rows:,} rows. AutoDS currently supports up to {MAX_DATAFRAME_ROWS:,} rows per session.",
        )
    if cols > MAX_DATAFRAME_COLUMNS:
        raise HTTPException(
            400,
            f"Dataset has {cols:,} columns. AutoDS currently supports up to {MAX_DATAFRAME_COLUMNS:,} columns per session.",
        )


def _ensure_allowed_remote_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(400, "URL must be an http:// or https:// address.")

    host = parsed.hostname
    if not host:
        raise HTTPException(400, "URL is missing a hostname.")

    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror as e:
        raise HTTPException(400, f"Could not resolve URL host: {e}")

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise HTTPException(400, "For safety, URL uploads must point to a public web address.")

    return url


def _read_csv_bytes(contents: bytes, sep: str | None = None) -> pd.DataFrame:
    kwargs = {"sep": sep} if sep else {}
    try:
        return pd.read_csv(io.BytesIO(contents), **kwargs)
    except pd.errors.ParserError:
        # Retry with Python's delimiter sniffer for files that are not comma CSVs.
        return pd.read_csv(io.BytesIO(contents), sep=None, engine="python")


def _friendly_parse_error(e: Exception, name: str) -> str:
    msg = str(e)
    if "openpyxl" in msg.lower() and name.lower().endswith(".xlsx"):
        return "Excel .xlsx support needs openpyxl installed. Run: pip install openpyxl"
    if "xlrd" in msg.lower() and name.lower().endswith(".xls"):
        return "Older .xls Excel files need xlrd installed. Run: pip install xlrd"
    if "tokenizing data" in msg.lower():
        return (
            f"{msg}. If this is a GitHub file, use the raw file URL or a normal "
            "github.com/.../blob/... link so AutoDS can convert it."
        )
    if "unsupported file type 'git'" in msg.lower() or name.lower().endswith(".git"):
        return (
            "That looks like a GitHub repository URL, not a dataset file. "
            "Open the CSV/Excel/JSON file inside the repo, click the file, then paste "
            "the github.com/.../blob/... link or the Raw link."
        )
    return msg


def _normalize_dataset_url(url: str) -> str:
    """Convert common share/view URLs into direct downloadable dataset URLs."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()

    if host == "github.com":
        parts = parsed.path.strip("/").split("/")
        if parsed.path.endswith(".git") or len(parts) <= 2 or (len(parts) >= 3 and parts[2] == "tree"):
            raise HTTPException(
                400,
                "That GitHub link points to a repository or folder, not a dataset file. "
                "Open a specific file such as .csv, .xlsx, .json, .tsv, .parquet, or .sqlite, "
                "then paste its github.com/.../blob/... link or Raw link.",
            )
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _, branch = parts[:4]
            path = "/".join(parts[4:])
            return urllib.parse.urlunparse((
                parsed.scheme or "https",
                "raw.githubusercontent.com",
                f"/{owner}/{repo}/{branch}/{path}",
                "",
                parsed.query,
                "",
            ))
        if len(parts) >= 5 and parts[2] == "raw":
            owner, repo, _, branch = parts[:4]
            path = "/".join(parts[4:])
            return urllib.parse.urlunparse((
                parsed.scheme or "https",
                "raw.githubusercontent.com",
                f"/{owner}/{repo}/{branch}/{path}",
                "",
                parsed.query,
                "",
            ))

    return url


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataframe memory usage by converting object columns to category
    and downcasting numeric types for faster processing."""
    for col in df.columns:
        col_type = df[col].dtype
        
        # Convert object/string columns to category if low cardinality
        if col_type == 'object' or str(col_type).startswith('str'):
            num_unique = df[col].nunique()
            num_total = len(df[col])
            # Convert to category if less than 50% unique values and more than 10 rows
            if num_total > 10 and num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
        
        # Downcast integer columns
        elif col_type in ['int64', 'int32']:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Downcast float columns
        elif col_type in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


def _df_from_bytes(contents: bytes, name: str) -> pd.DataFrame:
    n = name.lower()
    if n.endswith(".csv"):
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(io.BytesIO(contents), sep=sep,
                                 on_bad_lines="skip", encoding_errors="replace")
                if df.shape[1] > 1:
                    return _optimize_dtypes(df)
            except Exception:
                continue
        df = pd.read_csv(io.BytesIO(contents), on_bad_lines="skip")
        return _optimize_dtypes(df)
    if n.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(contents))
        return _optimize_dtypes(df)
    if n.endswith(".geojson"):
        try:
            geojson = json.loads(contents)
            features = geojson.get("features", [])
            if not features:
                raise ValueError("GeoJSON file contains no features.")
            rows = []
            for i, feature in enumerate(features):
                props = dict(feature.get("properties", {}))
                props["_feature_index"] = i          # used by choropleth to join back
                feature["properties"] = props
                geom  = feature.get("geometry", {})
                props["_geometry_type"] = geom.get("type", "Unknown")
                rows.append(props)
            df = pd.DataFrame(rows)
            return _optimize_dtypes(df), geojson     # return tuple so upload handler can store raw
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid GeoJSON: {e}")
    if n.endswith(".json"):
        df = pd.read_json(io.BytesIO(contents))
        return _optimize_dtypes(df)
    if n.endswith((".tsv", ".txt")):
        df = pd.read_csv(io.BytesIO(contents), sep="\t", on_bad_lines="skip")
        return _optimize_dtypes(df)
    if n.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(contents))
    if n.endswith(".db") or n.endswith(".sqlite") or n.endswith(".sqlite3"):
        # SQLite: return the first table found
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            f.write(contents); tmp = f.name
        try:
            con = sqlite3.connect(tmp)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
            if tables.empty:
                raise ValueError("No tables found in this SQLite file.")
            first_table = tables["name"].iloc[0]
            df = pd.read_sql(f"SELECT * FROM '{first_table}'", con)
            con.close()
        finally:
            os.unlink(tmp)
        return _optimize_dtypes(df)
    raise HTTPException(400, f"Unsupported file type '{name.split('.')[-1]}'. Supported: CSV, Excel, JSON, GeoJSON, TSV, Parquet, SQLite.")


def _session_response(session, df):
    return {
        "session_id": session.id,
        "filename":   session.filename,
        "columns":    list(df.columns),
        "shape":      {"rows": df.shape[0], "columns": df.shape[1]},
        "has_geojson": bool(getattr(session, "geojson", None)),
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "data.csv"
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        result = _df_from_bytes(contents, name)
        df, raw_geojson = result if isinstance(result, tuple) else (result, None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {_friendly_parse_error(e, name)}")
    if df.empty:
        raise HTTPException(400, "Uploaded file has no rows.")
    _validate_dataframe_size(df)
    session = create_session(df, name)
    if raw_geojson:
        session.geojson = raw_geojson
    return _session_response(session, df)


class UrlUploadRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    name: str = Field(default="", max_length=120)


@app.post("/api/upload_url")
def upload_url(req: UrlUploadRequest):
    """Fetch a dataset from any public URL — CSV, Excel, JSON, TSV, or a Google Sheets share link."""
    url = req.url.strip()
    # Convert Google Sheets share URL to CSV export URL
    if "docs.google.com/spreadsheets" in url:
        # extract spreadsheet id and convert to export link
        import re
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
        if not m:
            raise HTTPException(400, "Could not parse Google Sheets URL.")
        sheet_id = m.group(1)
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    else:
        url = _normalize_dataset_url(url)
    url = _ensure_allowed_remote_url(url)
    try:
        req2 = urllib.request.Request(url, headers={"User-Agent": "AutoDS/1.0"})
        with urllib.request.urlopen(req2, timeout=15) as resp:
            length = resp.headers.get("Content-Length")
            if length and int(length) > MAX_REMOTE_BYTES:
                raise HTTPException(400, f"Remote file is too large. Maximum size is {MAX_REMOTE_BYTES // (1024 * 1024)} MB.")
            contents = resp.read()
            if len(contents) > MAX_REMOTE_BYTES:
                raise HTTPException(400, f"Remote file is too large. Maximum size is {MAX_REMOTE_BYTES // (1024 * 1024)} MB.")
            content_type = resp.headers.get("Content-Type", "")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not fetch URL: {e}")

    # Guess filename from URL or content-type
    name = req.name or url.split("?")[0].split("/")[-1] or "data"
    if "." not in name.split("/")[-1]:
        if "excel" in content_type or "spreadsheet" in content_type:
            name += ".xlsx"
        elif "json" in content_type:
            name += ".json"
        else:
            name += ".csv"

    # Validate that we got actual data, not an error page
    if name.endswith('.json'):
        try:
            json.loads(contents[:200])
        except json.JSONDecodeError:
            raise HTTPException(
                400,
                "The URL did not return valid JSON. This often happens with GitHub URLs when rate-limited. "
                "Try using the raw file URL (raw.githubusercontent.com/...) or wait a moment and try again."
            )
    
    try:
        result = _df_from_bytes(contents, name)
        df, raw_geojson = result if isinstance(result, tuple) else (result, None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not parse data from URL: {_friendly_parse_error(e, name)}")
    if df.empty:
        raise HTTPException(400, "No rows found at that URL.")
    _validate_dataframe_size(df)
    session = create_session(df, name)
    if raw_geojson:
        session.geojson = raw_geojson
    return _session_response(session, df)


class PasteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_UPLOAD_BYTES)
    name: str = Field(default="pasted_data", max_length=120)
    separator: str = Field(default=",", max_length=8)   # "," for CSV, "\t" for TSV


@app.post("/api/upload_paste")
def upload_paste(req: PasteRequest):
    """Accept raw pasted CSV/TSV text."""
    try:
        sep = "\t" if req.separator in ("\\t", "\t", "tab") else req.separator
        df = pd.read_csv(io.StringIO(req.text.strip()), sep=sep)
    except Exception as e:
        raise HTTPException(400, f"Could not parse pasted text: {e}")
    if df.empty:
        raise HTTPException(400, "No rows found in pasted text.")
    _validate_dataframe_size(df)
    session = create_session(df, req.name + ".csv")
    return _session_response(session, df)


class DbRequest(BaseModel):
    connection_string: str = Field(min_length=1, max_length=2048)   # e.g. postgresql://user:pass@host/db
    query: str = Field(min_length=1, max_length=20_000)             # SQL SELECT query
    name: str = Field(default="db_query", max_length=120)

    @field_validator("query")
    @classmethod
    def only_select_queries(cls, value: str) -> str:
        query = value.strip()
        lowered = query.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise ValueError("Only SELECT queries are supported.")
        if ";" in query.rstrip(";"):
            raise ValueError("Only one SQL statement is allowed.")
        return query.rstrip(";")


@app.post("/api/upload_db")
def upload_db(req: DbRequest):
    """Run a SQL SELECT on a Postgres or MySQL database and load the result."""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(req.connection_string, connect_args={"connect_timeout": 10})
        with engine.connect() as con:
            df = pd.read_sql(text(req.query), con)
    except ImportError:
        raise HTTPException(500, "sqlalchemy is not installed. Run: pip install sqlalchemy psycopg2-binary")
    except Exception as e:
        raise HTTPException(400, f"Database error: {e}")
    if df.empty:
        raise HTTPException(400, "Query returned no rows.")
    _validate_dataframe_size(df)
    session = create_session(df, req.name + ".csv")
    return _session_response(session, df)


# ---------- EDA ----------

@app.get("/api/eda/{session_id}")
def get_eda(session_id: str):
    session = _get_session_or_404(session_id)
    profile = eda.profile_dataframe(session.df)
    profile["narrative"] = narrate.narrate_eda(profile)
    profile["can_undo"] = session.can_undo()
    return profile


@app.get("/api/pca/{session_id}")
def get_pca(session_id: str, color_col: str | None = None):
    """
    Run PCA on the session's current dataframe and return:
    - whether PCA is useful (feasible, verdict)
    - explained variance per component
    - how many components needed for 80/90/95% threshold
    - 2D scatter of the first two principal components
    - column loadings (which original columns drive each PC)
    Optional: pass ?color_col=colname to colour the 2D scatter by a categorical column.
    """
    session = _get_session_or_404(session_id)
    result = pca_analysis.analyse(session.df, color_col=color_col)
    # Cache it on the session — this is also the flag the downloaded HTML
    # report uses to decide whether to show a PCA section at all, so it only
    # appears if the user actually ran PCA from the app.
    session.pca_result = result
    return result


# ---------- Clean ----------

class CleanRequest(BaseModel):
    drop_duplicates: bool = True
    fill_missing_numeric: Literal["median", "mean", "zero", "none"] = "median"
    fill_missing_categorical: Literal["mode", "unknown", "none"] = "mode"
    drop_high_missing_cols: bool = True
    strip_whitespace: bool = True
    drop_constant_cols: bool = True
    fix_mixed_types: bool = True
    fix_column_names: bool = True
    remove_outliers: bool = True


@app.post("/api/clean/{session_id}")
def clean_data(session_id: str, req: CleanRequest):
    session = _get_session_or_404(session_id)
    session.snapshot_before_change()
    clean_options = req.model_dump()
    cleaned_df, log = clean_module.clean_dataframe(session.df, clean_options)
    session.df = cleaned_df
    session.clear_current_training()
    session.save_to_disk()

    # Merge the Clean Now log with any chat-cleaning already applied so the
    # report shows the full picture instead of wiping prior chat changes.
    if not log:
        log = ["Clean Now found nothing new to fix — your data is already in good shape"
               + (" from the chat cleaning you already applied." if session.chat_clean_log else ".")]
    if session.chat_clean_log:
        chat_summary = [f"Previously applied via chat assistant ({len(session.chat_clean_log)} command(s)):"]
        chat_summary += [f"  • {entry.get('command', '')}: {entry.get('message', '')}"
                         for entry in session.chat_clean_log[-10:]]
        log = chat_summary + [""] + log

    session.cleaning_log = log
    session.last_clean_options = clean_options

    profile = eda.profile_dataframe(session.df)
    profile["cleaning_log"] = log
    profile["narrative"] = narrate.narrate_eda(profile)
    profile["can_undo"] = session.can_undo()
    profile["data_changed"] = True
    return profile


class CleanChatRequest(BaseModel):
    command: str = Field(min_length=1, max_length=500)


class DatabaseAnalysisRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    # Optional: bucket a numeric column into custom ranges (e.g. income into
    # "Under 40k" / "40k-80k" / ...) before running the grouped analysis —
    # mirrors a SQL "CASE WHEN ... THEN ... END" bucketing column.
    bin_column: str | None = Field(default=None, max_length=200)
    bin_edges: list[float] | None = Field(default=None)
    bin_labels: list[str] | None = Field(default=None)

    @field_validator("bin_edges")
    @classmethod
    def validate_bin_edges(cls, value):
        if value is None:
            return value
        if not (1 <= len(value) <= 8):
            raise ValueError("Provide between 1 and 8 range breakpoints.")
        if any(v != v or v in (float("inf"), float("-inf")) for v in value):  # NaN / inf check
            raise ValueError("Range breakpoints must be finite numbers.")
        if sorted(set(value)) != sorted(value) or list(value) != sorted(value):
            raise ValueError("Range breakpoints must be in strictly increasing order with no duplicates.")
        return value

    @field_validator("bin_labels")
    @classmethod
    def validate_bin_labels(cls, value):
        if value is None:
            return value
        cleaned = [str(v).strip() for v in value]
        if not (2 <= len(cleaned) <= 9):
            raise ValueError("Provide between 2 and 9 bucket labels.")
        if any(not v for v in cleaned):
            raise ValueError("Bucket labels cannot be empty.")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Bucket labels must be unique.")
        return cleaned


@app.post("/api/clean_chat/{session_id}")
def clean_chat_endpoint(session_id: str, req: CleanChatRequest):
    session = _get_session_or_404(session_id)
    new_df, message, table = clean_chat.run_command_with_table(session.df, req.command, original_df=session.original_df)

    # Special action: split rows into a brand-new session (new tab)
    if isinstance(message, dict) and message.get("__action__") == "split_to_tab":
        action = message
        subset_df = action["subset_df"]   # full DataFrame stored by clean_chat
        new_session = create_session(subset_df, action["tab_name"] + ".csv")
        conditions_label = action.get("conditions_label", "")
        return {
            "split_to_tab": True,
            "new_session_id": new_session.id,
            "tab_name": action["tab_name"],
            "rows_matched": action["rows_matched"],
            "total_rows": action["total_rows"],
            "columns": action["columns"],
            "preview": action["subset_records"],
            **eda.profile_dataframe(session.df),
                        "chat_message": (
                "✓ Created new tab '"
                + action["tab_name"]
                + "' with "
                + f"{action['rows_matched']:,} of {action['total_rows']:,} rows "
                + f"({conditions_label}). Original dataset unchanged."
            ),
                        "chat_history": session.chat_clean_log,
            "data_changed": False,
            "can_undo": session.can_undo(),
        }

    changed = new_df is not session.df
    if changed:   # something actually changed (including "reset") — make it undoable
        session.snapshot_before_change()
    session.df = new_df
    if changed:
        session.clear_current_training()
        session.save_to_disk()
    session.chat_clean_log.append({"command": req.command, "message": message})

    profile = eda.profile_dataframe(session.df)
    profile["narrative"] = narrate.narrate_eda(profile)
    profile["can_undo"] = session.can_undo()
    profile["chat_message"] = message
    profile["chat_table"] = table
    profile["chat_history"] = session.chat_clean_log
    profile["data_changed"] = changed
    return profile


@app.get("/api/clean_chat/{session_id}/help")
def clean_chat_help(session_id: str):
    _get_session_or_404(session_id)  # validate session exists
    return {"help": clean_chat.HELP_TEXT}


@app.post("/api/database_analysis/{session_id}")
def database_analysis(session_id: str, req: DatabaseAnalysisRequest):
    """Read-only SQL-style grouped analysis for the Data Cleaning panel."""
    session = _get_session_or_404(session_id)
    try:
        answer, table = clean_chat.run_database_analysis(
            session.df, req.question, req.bin_column, req.bin_edges, req.bin_labels,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "answer": answer,
        "table": table,
        "data_changed": False,
    }


class TypeChangeRequest(BaseModel):
    column: str = Field(min_length=1)
    dtype: Literal["integer", "float", "text", "category", "datetime", "date", "time", "boolean"]


class NotesRequest(BaseModel):
    notes: str = Field(default="", max_length=20000)


@app.post("/api/notes/{session_id}")
def save_notes(session_id: str, req: NotesRequest):
    """Free-form notes the user writes about a dataset. Kept on the session so
    the work report (.md) and HTML report can include them under a Notes section."""
    session = _get_session_or_404(session_id)
    session.notes = req.notes
    return {"ok": True, "notes": session.notes}


@app.get("/api/notes/{session_id}")
def get_notes(session_id: str):
    session = _get_session_or_404(session_id)
    return {"notes": getattr(session, "notes", "")}


def _convert_column_type(df: pd.DataFrame, column: str, dtype: str) -> tuple[pd.DataFrame, str]:
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found.")

    dtype = dtype.lower().strip()
    allowed = {"integer", "float", "text", "category", "datetime", "date", "time", "boolean"}
    if dtype not in allowed:
        raise HTTPException(400, f"Unsupported data type '{dtype}'.")

    out = df.copy()
    before = str(out[column].dtype)
    before_missing = int(out[column].isna().sum())

    # Common date formats including Kenyan/African conventions (DD/MM/YYYY, DD-MM-YYYY)
    DATE_FORMATS = [
        "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y",
        "%d %b %Y", "%d %B %Y", "%Y/%m/%d",
        "%d/%m/%y", "%d-%m-%y",
    ]

    def _parse_dates(series):
        """Try pandas auto-parse first, then fall back through common formats."""
        # Try dayfirst=True by default — matches DD/MM/YYYY used in Kenya
        result = pd.to_datetime(series, dayfirst=True, errors="coerce")
        if result.notna().sum() >= series.notna().sum() * 0.5:
            return result
        for fmt in DATE_FORMATS:
            try:
                candidate = pd.to_datetime(series, format=fmt, errors="coerce")
                if candidate.notna().sum() > result.notna().sum():
                    result = candidate
            except Exception:
                continue
        return result

    try:
        if dtype == "integer":
            converted = pd.to_numeric(out[column], errors="coerce")
            # errors="coerce" silently turns any value it can't parse into
            # NaN — including genuinely non-numeric text like "C536379" in a
            # column that's mostly real numbers. Left unchecked, that means
            # converting a column to integer can silently delete real data
            # with nothing but an easy-to-miss note, while a decimal value
            # in the same column hard-blocks the conversion below. Treat
            # unparseable text the same way: refuse rather than destroy it.
            # (Blank/whitespace-only cells are treated as already-missing,
            # not as data that would be destroyed.)
            had_real_value = out[column].notna() & (out[column].astype(str).str.strip() != "")
            unparseable = converted.isna() & had_real_value
            if bool(unparseable.any()):
                examples = out[column][unparseable].astype(str).unique()[:3]
                example_str = ", ".join(f"'{v}'" for v in examples)
                raise HTTPException(
                    400,
                    f"Column '{column}' has non-numeric value(s) (e.g. {example_str}) that can't become "
                    f"integers. Converting would silently turn them into missing data. Clean those values "
                    f"first, or convert to text/category instead.",
                )
            non_integer = converted.dropna() % 1 != 0
            if bool(non_integer.any()):
                raise HTTPException(
                    400,
                    f"Column '{column}' contains decimal values, so it cannot be safely converted to integer.",
                )
            out[column] = converted.astype("Int64")
        elif dtype == "float":
            converted = pd.to_numeric(out[column], errors="coerce")
            had_real_value = out[column].notna() & (out[column].astype(str).str.strip() != "")
            unparseable = converted.isna() & had_real_value
            if bool(unparseable.any()):
                examples = out[column][unparseable].astype(str).unique()[:3]
                example_str = ", ".join(f"'{v}'" for v in examples)
                raise HTTPException(
                    400,
                    f"Column '{column}' has non-numeric value(s) (e.g. {example_str}) that can't become "
                    f"numbers. Converting would silently turn them into missing data. Clean those values "
                    f"first, or convert to text/category instead.",
                )
            out[column] = converted.astype(float)
        elif dtype == "text":
            out[column] = out[column].astype("string")
        elif dtype == "category":
            out[column] = out[column].astype("category")
        elif dtype == "datetime":
            out[column] = _parse_dates(out[column])
        elif dtype == "date":
            # Parse to datetime first, then keep only the date part as a string
            # (pandas date objects don't have a single clean dtype, so we store
            # as a formatted string "YYYY-MM-DD" which is unambiguous and sorts correctly)
            parsed = _parse_dates(out[column])
            out[column] = parsed.dt.strftime("%Y-%m-%d").astype("string")
            out[column] = out[column].where(parsed.notna(), pd.NA)
        elif dtype == "time":
            # Parse as datetime then extract time-of-day as "HH:MM:SS" string
            parsed = _parse_dates(out[column])
            # If parsing yielded all-midnight (failed), try direct time extraction
            if parsed.isna().all() or (parsed.dt.hour == 0).all():
                # Try treating raw values as time strings directly
                try:
                    parsed = pd.to_datetime("2000-01-01 " + out[column].astype(str), errors="coerce")
                except Exception:
                    pass
            out[column] = parsed.dt.strftime("%H:%M:%S").astype("string")
            out[column] = out[column].where(parsed.notna(), pd.NA)
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not convert '{column}' to {dtype}: {e}")

    after = str(out[column].dtype)
    after_missing = int(out[column].isna().sum())
    introduced = after_missing - before_missing
    type_label = {"date": "date only", "time": "time only", "datetime": "date & time"}.get(dtype, dtype)
    note = f"Changed column '{column}' from {before} to {type_label}."
    if introduced > 0:
        note += f" {introduced:,} value(s) could not be converted and became missing."
    return out, note


@app.post("/api/change_type/{session_id}")
def change_column_type(session_id: str, req: TypeChangeRequest):
    session = _get_session_or_404(session_id)
    session.snapshot_before_change()
    session.df, log_entry = _convert_column_type(session.df, req.column, req.dtype)
    session.clear_current_training()
    session.cleaning_log.append(log_entry)
    session.save_to_disk()

    profile = eda.profile_dataframe(session.df)
    profile["cleaning_log"] = session.cleaning_log
    profile["narrative"] = narrate.narrate_eda(profile)
    profile["can_undo"] = session.can_undo()
    profile["data_changed"] = True
    return profile


@app.post("/api/undo/{session_id}")
def undo_last_change(session_id: str):
    session = _get_session_or_404(session_id)
    if not session.undo():
        raise HTTPException(400, "Nothing to undo.")
    session.clear_current_training()

    profile = eda.profile_dataframe(session.df)
    profile["narrative"] = narrate.narrate_eda(profile)
    profile["can_undo"] = session.can_undo()
    profile["data_changed"] = True
    return profile


# ---------- Visualize ----------

def _ensure_named_geojson(session) -> bool:
    """
    Choropleth normally requires an uploaded .geojson file. Most datasets
    instead just have a plain column of country or US state *names*
    (e.g. "Egypt", "Texas") with no boundary shapes attached.

    If the session doesn't already have a GeoJSON, this tries to detect such
    a column by actually matching its values against bundled country/US
    state boundaries (backend/geo.py + backend/geo_data/). On a good match
    it attaches a synthetic '_feature_index' column (join key, same
    convention as an uploaded .geojson) and the matching boundaries, so the
    rest of the choropleth code path works unchanged.

    Returns True if geojson is available on the session after this call.
    """
    if session.geojson:
        return True
    col, level, matched = geo.find_name_column(session.df)
    if not col:
        return False
    session.df["_feature_index"] = session.df[col].map(matched)
    session.geojson = geo.load_geojson(level)
    session.geo_name_col = col
    session.geo_level = level
    return True


@app.get("/api/visualize/suggestions/{session_id}")
def visualize_suggestions(session_id: str):
    session = _get_session_or_404(session_id)
    try:
        _ensure_named_geojson(session)
        charts = viz.suggest_visuals(session.df, has_geojson=bool(session.geojson))
        if charts:
            session.last_visualization = charts[0]
        return {"charts": charts}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/column-values/{session_id}")
def column_values(session_id: str, col: str):
    """Return the distinct values of a column, so the frontend can offer a
    Power BI-style slicer (pick which values to compare in a chart)."""
    session = _get_session_or_404(session_id)
    df = session.df
    if col not in df.columns:
        raise HTTPException(400, f"Column '{col}' not found.")
    s = df[col].dropna()
    # Cap how many distinct values we ever send back — a free-text or ID
    # column can have hundreds of thousands of uniques, which would make
    # the slicer UI unusable (and the payload huge) for no real benefit.
    MAX_VALUES = 500
    counts = s.astype(str).value_counts()
    truncated = len(counts) > MAX_VALUES
    values = counts.head(MAX_VALUES).index.tolist()
    return {"col": col, "values": values, "truncated": truncated, "total_unique": int(len(counts))}


def _apply_compare_filter(df: pd.DataFrame, filter_col: str | None, filter_values: str | None) -> pd.DataFrame:
    """Restrict df to the rows whose filter_col matches one of the selected
    filter_values (comma-separated). Mirrors a Power BI slicer: picking two
    or more values lets the resulting chart compare just those groups.
    Returns df unchanged if no filter was supplied.
    """
    if not filter_col or filter_values is None:
        return df
    if filter_col not in df.columns:
        raise ValueError(f"Filter column '{filter_col}' not found.")
    wanted = [v for v in filter_values.split(",") if v != ""]
    if not wanted:
        return df
    # Compare as strings so numeric/date/category columns all match the
    # values the frontend sent (which came from the same string-cast list
    # used to populate the slicer checkboxes).
    mask = df[filter_col].astype(str).isin(wanted)
    filtered = df[mask]
    if filtered.empty:
        raise ValueError(
            f"No rows match the selected '{filter_col}' values — pick at least one value that exists in the data."
        )
    return filtered


@app.get("/api/visualize/{session_id}")
def visualize_custom(session_id: str, x: str, chart_type: str,
                     y: str | None = None, group: str | None = None,
                     x_min: float | None = None, x_max: float | None = None,
                     bar_limit: int | None = None, bin_width: float | None = None,
                     filter_col: str | None = None, filter_values: str | None = None):
    session = _get_session_or_404(session_id)
    try:
        df = _apply_compare_filter(session.df, filter_col, filter_values)

        if chart_type == "choropleth":
            _ensure_named_geojson(session)
            if not session.geojson:
                raise ValueError(
                    "Couldn't build a choropleth: no .geojson was uploaded, and no column "
                    "looked like recognizable country or US state names. Rename/check your "
                    "location column, or upload a .geojson."
                )
            if x not in df.columns:
                raise ValueError(f"Column '{x}' not found.")
            if not pd.api.types.is_numeric_dtype(df[x]):
                raise ValueError(f"'{x}' is not numeric. Choose a numeric value column for the choropleth.")

            # If we auto-attached geojson via name-matching, we already know
            # exactly which column matched — prefer that over the heuristic
            # guess below (which is for the manually-uploaded .geojson case,
            # where the geojson's own properties became the columns).
            name_col = getattr(session, "geo_name_col", None)
            if not name_col or name_col not in df.columns:
                skip_cols = {"_geometry_type", "_feature_index", x}
                text_cols = [
                    c for c in df.columns
                    if c not in skip_cols and not pd.api.types.is_numeric_dtype(df[c])
                ]
                name_hints = ("admin", "name", "country", "region", "county", "province", "state", "district", "city")

                def name_score(col: str) -> float:
                    hint = 3 if any(h in col.lower() for h in name_hints) else 0
                    non_empty = df[col].dropna().astype(str).str.strip()
                    non_empty = non_empty[non_empty != ""]
                    return hint + (len(non_empty) / max(len(df), 1))

                name_col = max(text_cols, key=name_score) if text_cols else None
            row_cols = ["_feature_index", x] + ([name_col] if name_col else [])
            rows = []
            for row in df[row_cols].to_dict("records"):
                cleaned = {}
                for key, value in row.items():
                    cleaned[key] = None if pd.isna(value) else value
                rows.append(cleaned)

            chart = {
                "type": "choropleth",
                "title": f"Choropleth - {x}",
                "reason": f"Colour each GeoJSON region by {x}.",
                "value_col": x,
                "name_col": name_col,
                "rows": rows,
                # "world" / "us_states" when auto-matched against our bundled
                # boundary files, or None for a manually-uploaded .geojson —
                # nb.py uses this to know which Plotly locationmode/lookup
                # table applies when exporting an interactive map.
                "level": getattr(session, "geo_level", None),
            }
            insight = narrate.explain_chart("choropleth", x, df)
            if insight:
                chart["insight"] = insight
            session.last_visualization = chart
            return chart

        chart = viz.chart_data(df, x, chart_type, y, group, x_min, x_max, bar_limit, bin_width)
        if filter_col:
            chart["filter_col"] = filter_col
            chart["filter_values"] = [v for v in (filter_values or "").split(",") if v != ""]
        insight = narrate.explain_chart(chart_type, x, df, y=y, group=group)
        if insight:
            chart["insight"] = insight
        session.last_visualization = chart
        return chart
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------- Simulate ----------

@app.post("/api/simulate/{session_id}")
def simulate_scenario(session_id: str, req: simulation_module.SimulationRequest):
    session = _get_session_or_404(session_id)
    try:
        return simulation_module.run_simulation(session.df, req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ---------- Download current data ----------

def _fmt_report_value(value) -> str:
    if value is pd.NA:
        return "missing"
    if pd.isna(value):
        return "missing"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value).replace("\n", " ").replace("\r", " ")


def _fmt_prediction_inputs(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No input values saved."
    row = rows[0]
    if not row:
        return "No input values saved."
    return "; ".join(f"{key}: {_fmt_report_value(value)}" for key, value in row.items())


def _compact_unsupervised_result(result: dict) -> dict:
    """Keep report-worthy unsupervised details without storing huge label arrays twice."""
    compact = dict(result)
    labels = compact.pop("labels", [])
    compact["n_rows_scored"] = len(labels)
    points = compact.pop("points", [])
    if points and not compact["n_rows_scored"]:
        compact["n_rows_scored"] = len(points)
    viz = compact.get("visualization")
    if isinstance(viz, dict) and "points" in viz:
        compact["visualization"] = {
            k: v for k, v in viz.items() if k != "points"
        }
        compact["n_visualized_points"] = len(viz.get("points") or [])
    if "anomaly_indices" in compact:
        compact["anomaly_indices_preview"] = compact["anomaly_indices"][:20]
        compact.pop("anomaly_indices", None)
    if "anomaly_scores" in compact:
        compact["anomaly_scores_preview"] = compact["anomaly_scores"][:10]
        compact.pop("anomaly_scores", None)
    return compact


def _has_cleaning_history(session) -> bool:
    return bool(session.cleaning_log or session.chat_clean_log)


def _build_work_report(session) -> str:
    profile = eda.profile_dataframe(session.df)
    narrative = narrate.narrate_eda(profile)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Determine how far the user has progressed
    has_cleaning = _has_cleaning_history(session)
    has_training  = bool(session.leaderboard)
    has_unsupervised = bool(session.unsupervised_results)
    has_predictions  = bool(session.saved_predictions)

    if has_training or has_predictions:
        stage = "Model Training & Prediction"
    elif has_unsupervised:
        stage = "Unsupervised Learning"
    elif has_cleaning:
        stage = "Data Cleaning"
    else:
        stage = "Exploratory Data Analysis"

    preview_title = "Cleaned Data — First 15 Rows" if has_cleaning else "Original Data — First 15 Rows"

    lines = [
        f"# AutoDS Work Report: {session.filename}",
        "",
        f"Generated: {generated_at}",
        f"Current stage: **{stage}**",
        "",
        "## Dataset",
        "",
        f"- Rows: {profile['shape']['rows']:,}",
        f"- Columns: {profile['shape']['columns']:,}",
        f"- Duplicate rows: {profile['duplicate_rows']:,}",
        f"- Missing cells: {profile['total_missing_cells']:,} of {profile['total_cells']:,}",
        "",
    ]

    if getattr(session, "notes", "").strip():
        lines.extend(["## Notes", ""])
        lines.extend(session.notes.strip().splitlines())
        lines.append("")

    lines.extend(["## Summary", ""])

    lines.extend(f"- {note}" for note in narrative)
    lines.extend(["", "## Columns", ""])

    for col in profile["columns"]:
        line = (
            f"- `{col['name']}`: {col['type']} ({col['dtype']}), "
            f"{col['unique']:,} unique, {col['missing']:,} missing ({col['missing_pct']}%)"
        )
        stats = col.get("stats")
        if stats:
            line += (
                f", mean {_fmt_report_value(stats.get('mean'))}, "
                f"min {_fmt_report_value(stats.get('min'))}, "
                f"max {_fmt_report_value(stats.get('max'))}"
            )
        top_values = col.get("top_values")
        if top_values:
            top = "; ".join(f"{v['value']}: {v['count']:,}" for v in top_values[:5])
            line += f", top values: {top}"
        lines.append(line)

    corr = profile.get("correlation", {})
    if corr.get("matrix") and len(corr.get("columns", [])) >= 2:
        pairs = []
        corr_cols = corr["columns"]
        matrix = corr["matrix"]
        for i, a in enumerate(corr_cols):
            for j in range(i + 1, len(corr_cols)):
                pairs.append((a, corr_cols[j], matrix[i][j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        lines.extend(["", "## Strongest Correlations", ""])
        for a, b, val in pairs[:10]:
            lines.append(f"- `{a}` vs `{b}`: {val:+.3f}")

    if _has_cleaning_history(session):
        lines.extend(["", "## Cleaning Log", ""])
        lines.extend(f"- {entry}" for entry in session.cleaning_log)
        lines.extend(f"- {entry.get('command', '')}: {entry.get('message', '')}" for entry in session.chat_clean_log)

    if session.leaderboard:
        lines.extend(["", "## Model Training", ""])
        lines.append(f"- Target column: `{session.target}`")
        lines.append(f"- Problem type: {session.problem_type.title() if session.problem_type else 'Unknown'}")
        lines.append(f"- Best model: **{session.best_model_name}**")
        lines.append(f"- Features used: {', '.join(f'`{c}`' for c in session.feature_columns) if session.feature_columns else 'all columns'}")
        lines.append("")
        lines.append("| Rank | Model | Metrics |")
        lines.append("| --- | --- | --- |")
        for i, row in enumerate(session.leaderboard, 1):
            if row.get("error"):
                metrics = f"failed: {row['error']}"
            else:
                metrics = ", ".join(f"{k}: {_fmt_report_value(v)}" for k, v in row.get("metrics", {}).items())
            marker = " ⭐" if row.get("model") == session.best_model_name else ""
            lines.append(f"| {i} | {row.get('model', 'model')}{marker} | {metrics} |")

    if session.saved_predictions:
        lines.extend(["", "## Predictions", ""])
        for prediction in session.saved_predictions.values():
            outputs = ", ".join(_fmt_report_value(v) for v in prediction.get("predictions", [])) or "missing"
            lines.append(f"- **Target:** `{prediction.get('target', 'Prediction')}` → {outputs}")
            lines.append(
                f"  - Model: {prediction.get('source_name', 'Current training')} "
                f"({prediction.get('model_name', 'model')})"
            )
            if prediction.get("created_at"):
                lines.append(f"  - Saved: {prediction['created_at']}")
            lines.append(f"  - Inputs: {_fmt_prediction_inputs(prediction.get('inputs', []))}")
            if prediction.get("narrative"):
                lines.append(f"  - Note: {prediction['narrative']}")

    if session.unsupervised_results:
        lines.extend(["", "## Unsupervised Learning", ""])
        u = session.unsupervised_results

        preprocessing = (u.get("suggestions") or {}).get("preprocessing") or {}
        if preprocessing:
            features = preprocessing.get("features_used") or []
            lines.append(f"**Preprocessing:** {preprocessing.get('scaling', 'StandardScaler')} scaling on {len(features):,} numeric feature(s)")
            if features:
                lines.append(f"- Features: {', '.join(f'`{c}`' for c in features[:15])}{'...' if len(features) > 15 else ''}")
            lines.append("")

        cluster_analysis = u.get("cluster_analysis")
        if cluster_analysis:
            lines.append("**Cluster Number Analysis**")
            lines.append(
                f"- Best silhouette K: {cluster_analysis.get('best_silhouette_k', '?')}; "
                f"best Davies-Bouldin K: {cluster_analysis.get('best_db_k', '?')}; "
                f"elbow K: {cluster_analysis.get('elbow_k', '?')}; "
                f"best Calinski-Harabasz K: {cluster_analysis.get('best_ch_k', '?')}"
            )
            rows_k = cluster_analysis.get("k_analysis") or []
            if rows_k:
                lines.append("")
                lines.append("| K | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz | Votes |")
                lines.append("| --- | --- | --- | --- | --- | --- |")
                for row in rows_k:
                    lines.append(
                        f"| {row.get('k')} | {_fmt_report_value(row.get('inertia'))} "
                        f"| {_fmt_report_value(row.get('silhouette'))} "
                        f"| {_fmt_report_value(row.get('davies_bouldin'))} "
                        f"| {_fmt_report_value(row.get('calinski_harabasz'))} "
                        f"| {row.get('votes', '')} |"
                    )
            lines.append("")

        clustering = u.get("clustering")
        if clustering:
            method = clustering.get("selected_method") or clustering.get("method", "Clustering")
            lines.append(f"**{method}** (chosen method)")
            reason = clustering.get("selection_reason", "")
            if reason:
                lines.append(f"- {reason}")
            if clustering.get("n_rows_scored"):
                lines.append(f"- Rows scored: {int(clustering['n_rows_scored']):,}")
            sizes = clustering.get("cluster_sizes") or {}
            if sizes:
                lines.append("")
                lines.append("| Cluster | Rows |")
                lines.append("| --- | --- |")
                for cname, count in sizes.items():
                    lines.append(f"| {cname} | {int(count):,} |")
            metrics = clustering.get("metrics") or {}
            if metrics:
                lines.append("")
                lines.append("Metrics: " + ", ".join(f"{k}: {_fmt_report_value(v)}" for k, v in metrics.items()))
            lines.append("")

        anomaly = u.get("anomaly")
        if anomaly:
            lines.append(f"**{anomaly.get('method', 'Anomaly Detection')}**")
            lines.append(f"- Anomalies found: {int(anomaly.get('n_outliers', 0)):,} ({_fmt_report_value(anomaly.get('outlier_percentage', 0))}% of data)")
            lines.append(f"- Normal rows: {int(anomaly.get('n_normal', 0)):,}")
            lines.append("")

        reduction = u.get("reduction")
        if reduction:
            lines.append(f"**{reduction.get('method', 'Dimensionality Reduction')}**")
            lines.append(f"- Components: {reduction.get('n_components', '?')}")
            lines.append(f"- Points scored: {int(reduction.get('n_rows_scored', 0)):,}")
            ev = reduction.get("explained_variance") or []
            if ev:
                lines.append("- Explained variance: " + ", ".join(f"PC{i+1}: {_fmt_report_value(v)}%" for i, v in enumerate(ev)))
            lines.append("")

        association = u.get("association")
        if association:
            lines.append(f"**{association.get('method', 'Association Rules')}**")
            lines.append(f"- Rules found: {int(association.get('n_rules', 0)):,}")
            lines.append(f"- Min support: {_fmt_report_value(association.get('min_support'))} | Min confidence: {_fmt_report_value(association.get('min_confidence'))}")
            rules = association.get("rules") or []
            if rules:
                lines.append("")
                lines.append("| Antecedents | Consequents | Support | Confidence | Lift |")
                lines.append("| --- | --- | --- | --- | --- |")
                for rule in rules[:10]:
                    ant = ", ".join(str(x) for x in (rule.get("antecedents") or []))
                    con = ", ".join(str(x) for x in (rule.get("consequents") or []))
                    lines.append(f"| {ant} | {con} | {_fmt_report_value(rule.get('support'))} | {_fmt_report_value(rule.get('confidence'))} | {_fmt_report_value(rule.get('lift'))} |")
            lines.append("")

    lines.extend(["", f"## {preview_title}", ""])
    preview = eda.safe_preview(session.df, 15)
    if preview:
        headers = list(preview[0].keys())
        lines.append("| " + " | ".join(map(str, headers)) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in preview:
            vals = [_fmt_report_value(row.get(h)).replace("|", "\\|") for h in headers]
            lines.append("| " + " | ".join(vals) + " |")
    else:
        lines.append("No preview rows available.")

    lines.append("")
    return "\n".join(lines)


@app.get("/api/geojson/{session_id}")
def get_geojson(session_id: str):
    """Return the raw GeoJSON stored on the session (uploaded .geojson file)."""
    session = get_session(session_id)
    if not session.geojson:
        raise HTTPException(404, "No GeoJSON available for this session.")
    return session.geojson


@app.get("/api/download_data/{session_id}")
def download_data(session_id: str, fmt: str = "csv"):
    session = _get_session_or_404(session_id)
    df = session.df
    name = session.filename.rsplit(".", 1)[0]

    if fmt == "excel":
        # Excel (.xlsx) sheets have a hard limit of 1,048,576 rows. Beyond
        # that the file can't be written at all — reject early with a clear
        # message instead of burning memory/time on a write that will fail
        # partway through anyway.
        EXCEL_ROW_LIMIT = 1_048_575  # leaves room for the header row
        if len(df) > EXCEL_ROW_LIMIT:
            raise HTTPException(
                400,
                f"This dataset has {len(df):,} rows, which is more than Excel's "
                f"limit of {EXCEL_ROW_LIMIT:,} rows per sheet. Please download as "
                "CSV instead, or filter/split the dataset into a smaller subset first.",
            )
        buf = io.BytesIO()
        # xlsxwriter's constant_memory mode streams rows to the output as it
        # writes instead of building the whole workbook as Python objects in
        # RAM first (which is what the default openpyxl engine does) — this
        # is the main fix for large exports using up all available memory.
        with pd.ExcelWriter(buf, engine="xlsxwriter", engine_kwargs={"options": {"constant_memory": True}}) as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={name}_autods.xlsx"},
        )

    # CSV: write in row-chunks and stream each chunk out as it's produced,
    # instead of building the entire CSV as one giant string in memory first.
    # This keeps peak memory roughly constant no matter how large the dataset is.
    CHUNK_ROWS = 50_000

    def _csv_chunks():
        header_buf = io.StringIO()
        df.iloc[0:0].to_csv(header_buf, index=False)
        yield header_buf.getvalue().encode("utf-8")
        for start in range(0, len(df), CHUNK_ROWS):
            chunk_buf = io.StringIO()
            df.iloc[start:start + CHUNK_ROWS].to_csv(chunk_buf, index=False, header=False)
            yield chunk_buf.getvalue().encode("utf-8")

    return StreamingResponse(
        _csv_chunks(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={name}_autods.csv"},
    )


@app.get("/api/download_work/{session_id}")
def download_work(session_id: str):
    session = _get_session_or_404(session_id)
    report = _build_work_report(session).encode("utf-8")
    buf = io.BytesIO(report)
    name = session.filename.rsplit(".", 1)[0]
    return StreamingResponse(
        buf,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={name}_autods_work_report.md"},
    )


class ReportChartsRequest(BaseModel):
    charts: list = []   # chart objects the user built in "Build a custom chart",
                         # in the same shape returned by /api/visualize


@app.post("/api/stage_charts/{session_id}")
def stage_charts(session_id: str, req: ReportChartsRequest):
    """
    Save the frontend's full chart history on the session so the plain GET
    .ipynb/.html downloads below can include every visualization the user
    built — not just the most recent one.

    This exists because a POST-with-a-blob-response download is unreliable
    on several mobile browsers (the response never renders/downloads after
    being opened in a new tab). The plain GET downloads use a real
    Content-Disposition-driven top-level navigation instead, which is the
    same reliable mechanism the working CSV/Excel/.md downloads already
    use — it just can't carry a POST body, hence staging the charts here
    first, then navigating to the GET endpoint.
    """
    session = _get_session_or_404(session_id)
    session.staged_charts = req.charts or []
    return {"ok": True, "charts_staged": len(session.staged_charts)}


@app.get("/api/download_html/{session_id}")
def download_html(session_id: str):
    session = _get_session_or_404(session_id)
    report = _build_html_report(session, extra_charts=session.staged_charts).encode("utf-8")
    buf = io.BytesIO(report)
    name = session.filename.rsplit(".", 1)[0]
    return StreamingResponse(
        buf,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={name}_autods_report.html"},
    )


@app.post("/api/download_html/{session_id}")
def download_html_with_custom_charts(session_id: str, req: ReportChartsRequest):
    session = _get_session_or_404(session_id)
    report = _build_html_report(session, extra_charts=req.charts).encode("utf-8")
    buf = io.BytesIO(report)
    name = session.filename.rsplit(".", 1)[0]
    return StreamingResponse(
        buf,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={name}_autods_report.html"},
    )


# ---------- Ask ----------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    # "clean" = sent from the Clean tab's chat, "assistant" = the floating
    # "Ask about your data" panel. A few data-modifying actions (like
    # creating a new column) are only meant for the Clean chat.
    source: str = "assistant"


@app.post("/api/ask/{session_id}")
def ask(session_id: str, req: AskRequest):
    session = _get_session_or_404(session_id)

    code_answer = assistant.answer_code_request(
        session.df,
        req.question,
        filename=session.filename,
        cleaning_log=session.cleaning_log,
        chat_clean_log=session.chat_clean_log,
        clean_options=session.last_clean_options,
        visualization=session.last_visualization,
        unsupervised_results=session.unsupervised_results,
    )
    if code_answer:
        return {"answer": code_answer, "table": None, "action": "show_code", "modified": False}
    
    # First try to execute an action
    action_result = assistant.execute_action(session.df, req.question, source=req.source)
    
    if action_result["success"] and action_result["modified_df"] is not None:
        # Apply the modification to the session
        session.snapshot_before_change()
        session.df = action_result["modified_df"]
        session.clear_current_training()
        
        # Get updated profile
        profile = eda.profile_dataframe(session.df)
        profile["narrative"] = narrate.narrate_eda(profile)
        profile["can_undo"] = session.can_undo()
        profile["data_changed"] = True
        
        return {
            "answer": action_result["message"],
            "table": action_result.get("table"),
            "action": action_result["action"],
            "profile": profile,
            "modified": True,
        }
    if action_result["success"]:
        return {
            "answer": action_result["message"],
            "table": action_result.get("table"),
            "action": action_result["action"],
            "modified": False,
        }
    
    # Otherwise, just answer the question
    answer = assistant.answer_question(session.df, req.question)
    table = assistant.answer_question_table(session.df, req.question)
    return {"answer": answer, "table": table, "action": None, "modified": False}


# ---------- Train ----------

class TrainRequest(BaseModel):
    target: str = Field(min_length=1)
    use_pca: bool = False


# session_id -> {"process", "manager", "progress", "result_queue", "target"}
# Training runs in its OWN OS process (not a thread) so it can actually be
# killed on cancel -- scikit-learn/xgboost .fit() calls can't be interrupted
# cooperatively from within the same process once they've started.
TRAIN_JOBS: dict = {}


def _train_worker(df, target: str, use_pca: bool, progress_list, result_queue):
    """Runs in a child process. Talks back to the parent only via progress_list/result_queue."""
    def _report(msg: str):
        progress_list.append(msg)

    try:
        problem_type, leaderboard, fitted, best_name, label_encoder = automl.train_all(
            df, target, use_pca=use_pca, progress_callback=_report
        )
        result_queue.put({
            "status": "done",
            "problem_type": problem_type,
            "leaderboard": leaderboard,
            "fitted": fitted,
            "best_name": best_name,
            "label_encoder": label_encoder,
        })
    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})


def _cleanup_train_job(session_id: str):
    job = TRAIN_JOBS.pop(session_id, None)
    if job is None:
        return
    if job["process"].is_alive():
        job["process"].terminate()
        job["process"].join(timeout=2)
        if job["process"].is_alive():
            job["process"].kill()
    job["manager"].shutdown()


@app.post("/api/train/{session_id}")
def train(session_id: str, req: TrainRequest):
    session = _get_session_or_404(session_id)
    if req.target not in session.df.columns:
        raise HTTPException(400, f"Column '{req.target}' not found.")

    # Automatically preserve the current training as a saved run before starting
    # a new one, so the user never loses a trained model when switching targets.
    if session.models and session.target and session.target != req.target:
        run_id = str(uuid.uuid4())
        session.saved_runs[run_id] = {
            "name": f"Auto-saved: {session.target}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "target": session.target,
            "problem_type": session.problem_type,
            "models": session.models,
            "leaderboard": session.leaderboard,
            "best_model_name": session.best_model_name,
            "label_encoder": session.label_encoder,
            "feature_columns": session.feature_columns,
        }

    # Starting a new run always replaces any previous one for this session.
    _cleanup_train_job(session_id)

    manager = mp.Manager()
    progress = manager.list(["Starting training...", "Preparing data and features..."])
    result_queue = manager.Queue()

    process = mp.Process(
        target=_train_worker,
        args=(session.df.copy(), req.target, req.use_pca, progress, result_queue),
        daemon=True,
    )
    process.start()

    TRAIN_JOBS[session_id] = {
        "process": process,
        "manager": manager,
        "progress": progress,
        "result_queue": result_queue,
        "target": req.target,
    }

    return {"status": "started"}


@app.get("/api/train/{session_id}/status")
def train_status(session_id: str):
    session = _get_session_or_404(session_id)
    job = TRAIN_JOBS.get(session_id)
    if job is None:
        return {"status": "idle", "messages": []}

    messages = list(job["progress"])

    if job["process"].is_alive():
        return {"status": "running", "messages": messages}

    # Process has exited -- see if it left a result behind before we clean it up.
    try:
        result = job["result_queue"].get_nowait()
    except Exception:
        result = None

    _cleanup_train_job(session_id)

    if result is None:
        # No result means it was cancelled (killed) or crashed without reporting.
        return {"status": "cancelled", "messages": messages}

    if result["status"] == "error":
        return {"status": "error", "messages": messages, "error": result["error"]}

    problem_type = result["problem_type"]
    leaderboard = result["leaderboard"]
    fitted = result["fitted"]
    best_name = result["best_name"]
    label_encoder = result["label_encoder"]
    target = job["target"]

    session.target = target
    session.problem_type = problem_type
    session.models = fitted
    session.leaderboard = leaderboard
    session.best_model_name = best_name
    session.label_encoder = label_encoder
    session.feature_columns = [c for c in session.df.columns if c != target]

    importance = []
    if best_name and best_name in fitted:
        X = session.df.drop(columns=[target])
        importance = automl.feature_importance(fitted[best_name], X)

    training_narrative = narrate.narrate_training(
        problem_type, target, leaderboard, best_name, importance
    )

    return {
        "status": "done",
        "messages": messages,
        "problem_type": problem_type,
        "target": target,
        "leaderboard": leaderboard,
        "best_model": best_name,
        "feature_importance": importance,
        "feature_columns": session.feature_columns,
        "narrative": training_narrative,
    }


@app.post("/api/train/{session_id}/cancel")
def cancel_train(session_id: str):
    job = TRAIN_JOBS.get(session_id)
    if job is None:
        return {"status": "idle"}
    _cleanup_train_job(session_id)
    return {"status": "cancelled"}


def _run_metric_summary(problem_type: str, leaderboard: list, best_name: str) -> dict | None:
    """Pull out the headline metric (accuracy or R²) for the best model, for display in run lists."""
    for r in leaderboard:
        if r.get("model") == best_name and "metrics" in r:
            key = "accuracy" if problem_type == "classification" else "r2"
            value = r["metrics"].get(key)
            return {"key": key, "value": value} if value is not None else None
    return None


def _run_summary(run_id: str, run: dict) -> dict:
    return {
        "id": run_id,
        "name": run["name"],
        "created_at": run["created_at"],
        "target": run["target"],
        "problem_type": run["problem_type"],
        "best_model": run["best_model_name"],
        "top_metric": _run_metric_summary(run["problem_type"], run["leaderboard"], run["best_model_name"]),
        "feature_columns": run["feature_columns"],
    }


class SaveRunRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


@app.post("/api/train_runs/{session_id}")
def save_train_run(session_id: str, req: SaveRunRequest):
    """Snapshot the current (most recently trained) models as a named run, so
    training a new target afterwards doesn't lose this one — it stays available
    for prediction under its own name."""
    session = _get_session_or_404(session_id)
    if not session.models:
        raise HTTPException(400, "Train a model first before saving a run.")

    run_id = str(uuid.uuid4())
    session.saved_runs[run_id] = {
        "name": req.name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target": session.target,
        "problem_type": session.problem_type,
        "models": session.models,
        "leaderboard": session.leaderboard,
        "best_model_name": session.best_model_name,
        "label_encoder": session.label_encoder,
        "feature_columns": session.feature_columns,
    }
    return {
        "saved": _run_summary(run_id, session.saved_runs[run_id]),
        "runs": [_run_summary(rid, r) for rid, r in session.saved_runs.items()],
    }


@app.get("/api/train_runs/{session_id}")
def list_train_runs(session_id: str):
    session = _get_session_or_404(session_id)
    return {"runs": [_run_summary(rid, r) for rid, r in session.saved_runs.items()]}


@app.delete("/api/train_runs/{session_id}/{run_id}")
def delete_train_run(session_id: str, run_id: str):
    session = _get_session_or_404(session_id)
    if run_id not in session.saved_runs:
        raise HTTPException(404, "Saved run not found.")
    del session.saved_runs[run_id]
    return {"runs": [_run_summary(rid, r) for rid, r in session.saved_runs.items()]}


# ---------- Predict ----------

class PredictRequest(BaseModel):
    rows: list[dict[str, object]] = Field(min_length=1, max_length=100)
    model_name: str | None = None  # defaults to best model
    run_id: str | None = None      # predict using a saved run instead of the current active training


def _jsonable_value(value):
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _prediction_summary(prediction_id: str, prediction: dict) -> dict:
    return {
        "id": prediction_id,
        "created_at": prediction["created_at"],
        "target": prediction["target"],
        "problem_type": prediction["problem_type"],
        "model_name": prediction["model_name"],
        "source_name": prediction["source_name"],
        "run_id": prediction["run_id"],
        "inputs": prediction["inputs"],
        "predictions": prediction["predictions"],
        "narrative": prediction["narrative"],
    }


@app.post("/api/predict/{session_id}")
def predict(session_id: str, req: PredictRequest):
    session = _get_session_or_404(session_id)

    if req.run_id:
        if req.run_id not in session.saved_runs:
            raise HTTPException(404, "Saved run not found.")
        run = session.saved_runs[req.run_id]
        models, problem_type, target = run["models"], run["problem_type"], run["target"]
        label_encoder, feature_columns = run["label_encoder"], run["feature_columns"]
        default_model_name = run["best_model_name"]
        source_name = run["name"]
    else:
        if not session.models:
            raise HTTPException(400, "No trained models for this session yet.")
        models, problem_type, target = session.models, session.problem_type, session.target
        label_encoder, feature_columns = session.label_encoder, session.feature_columns
        default_model_name = session.best_model_name
        source_name = "Current training"

    model_name = req.model_name or default_model_name
    if model_name not in models:
        raise HTTPException(400, f"Model '{model_name}' not found.")

    pipe = models[model_name]
    input_df = pd.DataFrame(req.rows)

    missing_cols = [c for c in feature_columns if c not in input_df.columns]
    for c in missing_cols:
        input_df[c] = None
    input_df = input_df[feature_columns]

    preds = pipe.predict(input_df)

    if problem_type == "classification" and label_encoder is not None:
        preds = label_encoder.inverse_transform(preds.astype(int))

    pred_value = preds[0].item() if hasattr(preds[0], "item") else preds[0]
    explanation = narrate.narrate_prediction(
        problem_type, target, pred_value, model_name, req.rows[0]
    )
    predictions = [_jsonable_value(p) for p in preds]
    prediction_id = str(uuid.uuid4())
    session.saved_predictions[prediction_id] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target": target,
        "problem_type": problem_type,
        "model_name": model_name,
        "source_name": source_name,
        "run_id": req.run_id,
        "inputs": req.rows,
        "predictions": predictions,
        "narrative": explanation,
    }

    return {
        "predictions": predictions,
        "narrative": explanation,
        "saved_prediction": _prediction_summary(prediction_id, session.saved_predictions[prediction_id]),
        "saved_predictions": [
            _prediction_summary(pid, p) for pid, p in session.saved_predictions.items()
        ],
    }


@app.get("/api/predictions/{session_id}")
def list_predictions(session_id: str):
    session = _get_session_or_404(session_id)
    return {
        "predictions": [
            _prediction_summary(pid, p) for pid, p in session.saved_predictions.items()
        ]
    }


@app.delete("/api/predictions/{session_id}/{prediction_id}")
def delete_prediction(session_id: str, prediction_id: str):
    session = _get_session_or_404(session_id)
    if prediction_id not in session.saved_predictions:
        raise HTTPException(404, "Saved prediction not found.")
    del session.saved_predictions[prediction_id]
    return {
        "predictions": [
            _prediction_summary(pid, p) for pid, p in session.saved_predictions.items()
        ]
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "sessions": len(SESSIONS),
        "limits": {
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_rows": MAX_DATAFRAME_ROWS,
            "max_columns": MAX_DATAFRAME_COLUMNS,
        },
    }


# ---------- Unsupervised Learning ----------

@app.get("/api/unsupervised/suggest/{session_id}")
def unsupervised_suggest(session_id: str):
    session = _get_session_or_404(session_id)
    result = unsupervised.suggest_unsupervised(session.df)
    result["preprocessing"] = unsupervised.preprocessing_summary(session.df)
    session.unsupervised_results["suggestions"] = result
    return result


@app.get("/api/unsupervised/suggest_clusters/{session_id}")
def unsupervised_suggest_clusters(session_id: str, max_clusters: int = 10):
    session = _get_session_or_404(session_id)
    try:
        result = unsupervised.suggest_clusters(session.df, max_clusters=max_clusters)
        session.unsupervised_results["cluster_analysis"] = _compact_unsupervised_result(result)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/unsupervised/cluster/{session_id}")
def unsupervised_cluster(session_id: str, req: dict):
    session = _get_session_or_404(session_id)
    method = req.get("method", "kmeans")
    try:
        if method == "auto":
            result = unsupervised.cluster_best(session.df, max_clusters=req.get("max_clusters", 10))
        elif method == "kmeans":
            result = unsupervised.cluster_kmeans(session.df, n_clusters=req.get("n_clusters", 3))
        elif method == "dbscan":
            result = unsupervised.cluster_dbscan(session.df, eps=req.get("eps", 0.5), min_samples=req.get("min_samples", 5))
        elif method == "hierarchical":
            result = unsupervised.cluster_hierarchical(session.df, n_clusters=req.get("n_clusters", 3), linkage=req.get("linkage", "ward"))
        else:
            raise HTTPException(400, f"Unknown clustering method: {method}")
        if len(result.get("labels", [])) == len(session.df):
            session.df["cluster_label"] = result["labels"]
        session.unsupervised_results["clustering"] = _compact_unsupervised_result(result)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/unsupervised/anomaly/{session_id}")
def unsupervised_anomaly(session_id: str, req: dict):
    session = _get_session_or_404(session_id)
    method = req.get("method", "isolation_forest")
    try:
        if method == "isolation_forest":
            result = unsupervised.detect_anomalies_isolation_forest(session.df, contamination=req.get("contamination", 0.1))
        elif method == "lof":
            result = unsupervised.detect_anomalies_lof(session.df, contamination=req.get("contamination", 0.1), n_neighbors=req.get("n_neighbors", 20))
        else:
            raise HTTPException(400, f"Unknown anomaly detection method: {method}")
        session.unsupervised_results["anomaly"] = _compact_unsupervised_result(result)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/unsupervised/reduce/{session_id}")
def unsupervised_reduce(session_id: str, req: dict):
    session = _get_session_or_404(session_id)
    method = req.get("method", "tsne")
    try:
        if method == "tsne":
            result = unsupervised.reduce_tsne(session.df, n_components=req.get("n_components", 2), perplexity=req.get("perplexity", 30.0))
        elif method == "pca":
            result = unsupervised.reduce_pca_advanced(session.df, n_components=req.get("n_components", 2))
        else:
            raise HTTPException(400, f"Unknown dimensionality reduction method: {method}")
        session.unsupervised_results["reduction"] = _compact_unsupervised_result(result)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/unsupervised/association/{session_id}")
def unsupervised_association(session_id: str, req: dict):
    session = _get_session_or_404(session_id)
    try:
        result = unsupervised.association_rules(
            session.df,
            min_support=req.get("min_support", 0.1),
            min_confidence=req.get("min_confidence", 0.5),
        )
        session.unsupervised_results["association"] = _compact_unsupervised_result(result)
        return result
    except ImportError as e:
        raise HTTPException(500, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------- Download Jupyter Notebook ----------

@app.get("/api/download_notebook/{session_id}")
def download_notebook(session_id: str):
    """Download a fully self-contained .ipynb with all AutoDS functions plus
    the current session data embedded as CSV. Includes the full chart
    history if it was staged first via /api/stage_charts (falls back to
    just the most recent chart, session.last_visualization, otherwise)."""
    session = _get_session_or_404(session_id)
    notebook_json = nb_module.build_notebook(session, charts=session.staged_charts)
    buf = io.BytesIO(notebook_json.encode("utf-8"))
    name = session.filename.rsplit(".", 1)[0]
    return StreamingResponse(
        buf,
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": f'attachment; filename="{name}_autods_notebook.ipynb"',
        },
    )


@app.post("/api/download_notebook/{session_id}")
def download_notebook_with_charts(session_id: str, req: ReportChartsRequest):
    """Same as the GET version, but takes the frontend's full chart history
    (customChartSpecs) so every visualization the user built gets its own
    cell in the notebook, not just the last one."""
    session = _get_session_or_404(session_id)
    notebook_json = nb_module.build_notebook(session, charts=req.charts)
    buf = io.BytesIO(notebook_json.encode("utf-8"))
    name = session.filename.rsplit(".", 1)[0]
    return StreamingResponse(
        buf,
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": f'attachment; filename="{name}_autods_notebook.ipynb"',
        },
    )


# ---------- Download trained model ----------

@app.get("/api/download_model/{session_id}")
def download_model(session_id: str, model_name: str | None = None, run_id: str | None = None):
    session = _get_session_or_404(session_id)

    if run_id:
        if run_id not in session.saved_runs:
            raise HTTPException(404, "Saved run not found.")
        run = session.saved_runs[run_id]
        models, label_encoder, feature_columns = run["models"], run["label_encoder"], run["feature_columns"]
        target, problem_type = run["target"], run["problem_type"]
        default_model_name = run["best_model_name"]
    else:
        models, label_encoder, feature_columns = session.models, session.label_encoder, session.feature_columns
        target, problem_type = session.target, session.problem_type
        default_model_name = session.best_model_name

    name = model_name or default_model_name
    if not name or name not in models:
        raise HTTPException(400, "No such trained model.")

    buf = io.BytesIO()
    pickle.dump({
        "pipeline": models[name],
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "target": target,
        "problem_type": problem_type,
    }, buf)
    buf.seek(0)
    safe_name = name.replace(" ", "_").lower()
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={safe_name}_model.pkl"},
    )


def _get_session_or_404(session_id: str):
    try:
        return get_session(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found. Please re-upload your file.")


@app.delete("/api/session/{session_id}")
def delete_session_endpoint(session_id: str):
    """Removes this session from memory and deletes its two on-disk CSV copies
    (current + original) from .sessions/. Called when a dataset tab is closed
    so storage doesn't grow forever."""
    delete_session(session_id)
    return {"ok": True}


@app.get("/api/progress/{session_id}")
def get_progress(session_id: str):
    session = _get_session_or_404(session_id)
    return {"messages": session.progress_messages}


# ---------- Filter / Split endpoint ----------

class FilterRequest(BaseModel):
    where_clause: str = Field(min_length=1, max_length=4000)   # pandas-style query string
    new_name: str = Field(default="filtered_subset", max_length=120)


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    """
    Convert a DataFrame slice to a list of dicts that's safe to return as
    JSON. Plain .to_dict(orient="records") leaves NaN/NaT/inf values as
    float('nan'), and FastAPI's JSON encoder raises
    'Out of range float values are not JSON compliant: nan' when it hits
    those. Swap them for None (-> JSON null) instead.
    """
    records = df.to_dict(orient="records")
    cleaned = []
    for row in records:
        cleaned_row = {}
        for key, value in row.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                cleaned_row[key] = None
            elif pd.isna(value):
                cleaned_row[key] = None
            else:
                cleaned_row[key] = value
        cleaned.append(cleaned_row)
    return cleaned


def _apply_filter(df: pd.DataFrame, where_clause: str) -> pd.DataFrame:
    """
    Run a filter using pandas .query().
    Supports expressions like:
        gender == 'Male'
        age > 30 and monthly_charges < 60
        churn == 'Yes' or tenure > 60
        city in ['Nairobi', 'Mombasa']
    Raises ValueError with a readable message on bad syntax.
    """
    try:
        result = df.query(where_clause, engine="python")
        return result.reset_index(drop=True)
    except Exception as exc:
        raise ValueError(f"Filter error: {exc}")


@app.post("/api/filter/{session_id}")
def filter_dataset(session_id: str, req: FilterRequest):
    """
    Apply a filter expression to the current dataframe and create a brand-new
    session from the matching rows. Returns a new session_id the frontend can
    open as an independent dataset tab.
    """
    session = _get_session_or_404(session_id)
    try:
        result_df = _apply_filter(session.df, req.where_clause)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    if result_df.empty:
        raise HTTPException(400, "The filter matched 0 rows. Try a less restrictive condition.")

    new_name = req.new_name.strip() or "filtered_subset"
    new_session = create_session(result_df, new_name + ".csv")

    return {
        "session_id": new_session.id,
        "name": new_name,
        "shape": {"rows": len(result_df), "columns": len(result_df.columns)},
        "columns": list(result_df.columns),
        "preview": _json_safe_records(result_df.head(5)),
    }


@app.post("/api/filter_preview/{session_id}")
def filter_preview(session_id: str, req: FilterRequest):
    """
    Dry-run: return the first 10 rows + row count for the filter expression
    without creating a new session.
    """
    session = _get_session_or_404(session_id)
    try:
        result_df = _apply_filter(session.df, req.where_clause)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return {
        "rows_matched": len(result_df),
        "total_rows": len(session.df),
        "preview": _json_safe_records(result_df.head(10)),
        "columns": list(result_df.columns),
    }


# ---------- Serve frontend ----------
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")