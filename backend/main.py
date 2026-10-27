import asyncio
import csv
import io
import html
import ipaddress
import logging
import math
import multiprocessing as mp
import os
import pickle
import re
import socket
import sqlite3
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Literal

import pandas as pd
import numpy as np
import xlsxwriter
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
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
from . import pivot as pivot_module
from . import unsupervised
from . import recommender
from . import forecast
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


# --- Bring-your-own-model import (see import_model below) -----------------
#
# Plain pickle.load() on a file someone else gave you is a code-execution
# risk: a pickle's opcode stream can reconstruct ANY class it names, and
# unpickling can run arbitrary code as a side effect (the classic exploit
# is a crafted __reduce__ that calls os.system). AutoDS's OWN model
# downloads are safe to re-load as-is (AutoDS wrote them seconds earlier),
# but a model from someone else gets no such trust.
#
# _RestrictedUnpickler only allows the file to reference classes/functions
# that live in the handful of libraries a trained model realistically
# needs (scikit-learn, numpy, scipy, pandas, the common external tree-
# boosting libraries, and plain containers) -- anything else (os,
# subprocess, builtins.eval/exec/__import__, a custom class from someone's
# notebook, ...) is refused before it's ever constructed.
class UnsafeModelFileError(Exception):
    pass


_ALLOWED_PICKLE_MODULE_ROOTS = (
    "sklearn", "numpy", "scipy", "pandas",
    "xgboost", "lightgbm", "catboost",
    "collections", "copyreg",
)
_ALLOWED_PICKLE_BUILTINS = {
    "list", "dict", "set", "frozenset", "tuple",
    "str", "int", "float", "bool", "bytes", "bytearray", "complex", "object",
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        root = module.split(".", 1)[0]
        if module == "builtins":
            if name not in _ALLOWED_PICKLE_BUILTINS:
                raise UnsafeModelFileError(f"blocked unsafe reference: builtins.{name}")
        elif root not in _ALLOWED_PICKLE_MODULE_ROOTS:
            raise UnsafeModelFileError(f"blocked unsafe reference: {module}.{name}")
        return super().find_class(module, name)


def _safe_unpickle_model(contents: bytes):
    """Unpickle an uploaded model file through the restricted allowlist
    above. Raises UnsafeModelFileError if the file references anything
    outside it, and lets ModuleNotFoundError propagate as-is (a reference
    to an allowed library that just isn't installed here, e.g. xgboost --
    distinct from a blocked/unsafe one, so callers can give a clearer
    error message for each case)."""
    return _RestrictedUnpickler(io.BytesIO(contents)).load()


def _extract_model_pipeline(obj):
    """An uploaded model file might be AutoDS's own download_model bundle
    (a dict with a "pipeline" key), some other dict shape using "model" or
    "estimator", or just the bare fitted estimator/pipeline itself. Returns
    the object to call .predict() on, or None if nothing predictable was
    found in it."""
    if hasattr(obj, "predict") and callable(getattr(obj, "predict", None)):
        return obj
    if isinstance(obj, dict):
        for key in ("pipeline", "model", "estimator"):
            candidate = obj.get(key)
            if candidate is not None and hasattr(candidate, "predict") and callable(candidate.predict):
                return candidate
    return None

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
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
    # Without this, a cross-origin fetch() (e.g. the desktop app's webview
    # calling a separately-hosted API) can't read Content-Disposition on the
    # response at all -- response.headers.get(...) silently returns null, so
    # downloads fall back to a generic filename. Browsers never expose
    # response headers cross-origin unless the server explicitly allows it.
    expose_headers=["Content-Disposition"],
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


# Separators we auto-detect / accept, plus friendly aliases a user (or the
# frontend's separator dropdown) can send instead of the raw character.
_SEP_ALIASES = {
    "comma": ",",
    "semicolon": ";",
    "tab": "\t", "\\t": "\t",
    "pipe": "|",
    "colon": ":",
    "space": " ",
}
_SEP_CANDIDATES = [",", ";", "\t", "|", ":"]


def _normalize_sep(sep: str | None) -> str | None:
    """Turn a user-supplied separator (raw char or friendly name like 'tab') into
    the literal character pandas expects. Returns None if nothing was given."""
    if sep is None:
        return None
    sep = sep.strip()
    if not sep:
        return None
    return _SEP_ALIASES.get(sep.lower(), sep)


def _row_match_score(sample: str, sep: str) -> tuple[int, float]:
    """Parse `sample` with `sep` (treated as regex if it's a character class
    like '[,;]') and report (num_columns, fraction_of_rows_with_that_width).

    A low fraction means the file is "ragged" under this separator -- a sign
    that rows don't actually share one consistent delimiter.
    """
    try:
        df = pd.read_csv(
            io.StringIO(sample), sep=sep, engine="python",
            on_bad_lines="skip", nrows=200,
        )
    except Exception:
        return (1, 0.0)
    if df.shape[1] <= 1 or df.empty:
        return (df.shape[1], 0.0)
    # Re-parse keeping ragged lines (skip nothing) so we can see how many rows
    # actually had the "right" number of fields vs. got dropped/mangled.
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    if len(lines) < 2:
        return (df.shape[1], 1.0)
    header_fields = len(re.split(sep, lines[0])) if any(c in sep for c in "[](){}|") or len(sep) > 1 else lines[0].count(sep) + 1
    body = lines[1:51]
    if not body:
        return (df.shape[1], 1.0)
    matches = sum(1 for ln in body if (len(re.split(sep, ln)) if (any(c in sep for c in "[](){}|") or len(sep) > 1) else ln.count(sep) + 1) == header_fields)
    return (df.shape[1], matches / len(body))


def _detect_separator(contents: bytes) -> str:
    """Guess the field separator of a delimited text file.

    Tries each common candidate on a sample of the file and keeps whichever
    yields the most columns with the most consistent row widths. If every
    single-character candidate produces "ragged" rows (a sign the file
    genuinely mixes delimiters -- e.g. some fields use ',' and others use ';'
    interchangeably), falls back to a combined regex that accepts any of the
    top candidates as a delimiter. Falls back to csv.Sniffer, then comma.
    """
    sample = contents[:65536].decode("utf-8", errors="replace")

    scored = []  # (cols, match_fraction, sep)
    for sep in _SEP_CANDIDATES:
        cols, frac = _row_match_score(sample, re.escape(sep))
        if cols > 1:
            scored.append((cols, frac, sep))

    if not scored:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|: ")
            return dialect.delimiter
        except Exception:
            return ","

    scored.sort(key=lambda t: (t[1], t[0]), reverse=True)
    best_cols, best_frac, best_sep = scored[0]
    if best_frac >= 0.9:
        return best_sep

    # No single character gives consistent rows -- the file likely mixes
    # delimiters. Try a regex character class over the top candidates (by
    # column count) and keep it only if it's actually more consistent.
    top_chars = sorted({s for _, _, s in scored[:3]})
    combo = "[" + re.escape("".join(top_chars)) + "]"
    combo_cols, combo_frac = _row_match_score(sample, combo)
    if combo_cols > 1 and combo_frac > best_frac:
        return combo

    return best_sep


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


def _read_delimited(contents: bytes, sep: str) -> pd.DataFrame:
    """pd.read_csv with the right engine: regex separators (e.g. the '[,;]'
    combo _detect_separator can return for mixed-delimiter files) only work
    with the python engine, which also tolerates ragged/bad lines better."""
    needs_python = len(sep) != 1
    return pd.read_csv(
        io.BytesIO(contents), sep=sep,
        engine="python" if needs_python else "c",
        on_bad_lines="skip", encoding_errors="replace",
    )


def _read_delimited_no_header(contents: bytes, sep: str) -> pd.DataFrame:
    """Split every line (there is no header row to anchor column count on) by
    `sep`, auto-naming columns Column_1, Column_2, ... up to the widest row.

    Used instead of pd.read_csv for the "no header" upload option because
    read_csv's C/python engines both infer the expected column count from a
    single reference row (the header, or row 1 if header=None) and then
    either pad or SILENTLY DROP every other row to match -- which badly
    mangles a file whose header and body don't share the same delimiter
    (e.g. a comma-separated header over pipe-separated data), turning most of
    the real fields into a throwaway index. Splitting line-by-line in plain
    Python sidesteps that entirely and just keeps whatever each row has.
    """
    text = contents.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    is_regex = len(sep) != 1
    rows = [re.split(sep, ln) if is_regex else ln.split(sep) for ln in lines]
    max_cols = max((len(r) for r in rows), default=1)
    rows = [r + [None] * (max_cols - len(r)) for r in rows]
    cols = [f"Column_{i + 1}" for i in range(max_cols)]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return df.replace({"": pd.NA})


def _df_from_bytes(contents: bytes, name: str, sep: str | None = None, has_header: bool = True) -> pd.DataFrame:
    n = name.lower()
    if n.endswith((".csv", ".txt")):
        # Honor an explicit separator (from the upload form or URL param) over
        # auto-detection, so a user can fix a wrongly-guessed delimiter. A
        # custom separator with more than one character (e.g. "[,;]") is
        # treated as a regex -- useful for files that genuinely mix delimiters.
        chosen_sep = _normalize_sep(sep) or _detect_separator(contents)
        if not has_header:
            # No row is trustworthy as "the header" (e.g. the actual header
            # uses a different separator than the data rows) -- split every
            # line positionally instead of letting read_csv infer width from
            # one row and silently drop/mangle the rest.
            return _optimize_dtypes(_read_delimited_no_header(contents, chosen_sep))
        try:
            df = _read_delimited(contents, chosen_sep)
            if df.shape[1] > 1 or sep:
                return _optimize_dtypes(df)
        except Exception:
            pass
        # Last-resort fallback: default comma parsing.
        df = pd.read_csv(io.BytesIO(contents), on_bad_lines="skip", encoding_errors="replace")
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
    if n.endswith(".tsv"):
        chosen_sep = _normalize_sep(sep) or "\t"
        if not has_header:
            return _optimize_dtypes(_read_delimited_no_header(contents, chosen_sep))
        df = _read_delimited(contents, chosen_sep)
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
async def upload(file: UploadFile = File(...), separator: str | None = Form(None), has_header: bool = Form(True)):
    name = file.filename or "data.csv"
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        result = _df_from_bytes(contents, name, sep=separator, has_header=has_header)
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
    separator: str | None = Field(default=None, max_length=8)
    has_header: bool = True


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
        result = _df_from_bytes(contents, name, sep=req.separator, has_header=req.has_header)
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
        sep = _normalize_sep(req.separator) or ","
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


# ---------- Merge / join two datasets ----------

class MergeRequest(BaseModel):
    left_session_id: str = Field(min_length=1)
    right_session_id: str = Field(min_length=1)
    left_on: list[str] = Field(min_length=1, max_length=10)
    right_on: list[str] = Field(min_length=1, max_length=10)
    how: Literal["inner", "left", "right", "outer"] = "inner"
    name: str = Field(default="", max_length=120)

    @field_validator("left_on", "right_on")
    @classmethod
    def no_blank_keys(cls, value: list[str]) -> list[str]:
        if any(not v.strip() for v in value):
            raise ValueError("Join key columns cannot be blank.")
        return value


@app.get("/api/merge_columns")
def merge_columns(left_session_id: str, right_session_id: str):
    """Return each side's column list, plus names present on both sides --
    used by the merge UI to populate key-column dropdowns and suggest matches."""
    left = _get_session_or_404(left_session_id)
    right = _get_session_or_404(right_session_id)
    left_cols = list(left.df.columns)
    right_cols = list(right.df.columns)
    shared = [c for c in left_cols if c in right_cols]
    return {
        "left_columns": left_cols,
        "right_columns": right_cols,
        "suggested_keys": shared,
    }


@app.post("/api/merge")
def merge_datasets(req: MergeRequest):
    """Merge (join) two already-uploaded datasets into a new dataset/session,
    mirroring pandas.merge. left_on/right_on may list several columns for a
    composite key; they're matched positionally (left_on[i] <-> right_on[i])."""
    if req.left_session_id == req.right_session_id:
        raise HTTPException(400, "Choose two different datasets to merge.")
    left = _get_session_or_404(req.left_session_id)
    right = _get_session_or_404(req.right_session_id)

    if len(req.left_on) != len(req.right_on):
        raise HTTPException(400, "left_on and right_on must list the same number of columns.")

    missing_left = [c for c in req.left_on if c not in left.df.columns]
    missing_right = [c for c in req.right_on if c not in right.df.columns]
    if missing_left:
        raise HTTPException(400, f"Column(s) not found in the left dataset: {', '.join(missing_left)}")
    if missing_right:
        raise HTTPException(400, f"Column(s) not found in the right dataset: {', '.join(missing_right)}")

    try:
        merged = pd.merge(
            left.df, right.df,
            left_on=req.left_on, right_on=req.right_on,
            how=req.how, suffixes=("_left", "_right"),
        )
    except Exception as e:
        raise HTTPException(400, f"Could not merge datasets: {e}")

    if merged.empty:
        raise HTTPException(
            400,
            f"The {req.how} join produced no rows. Double-check the join key(s) and join type "
            "(e.g. an inner join needs matching values on both sides).",
        )

    _validate_dataframe_size(merged)
    name = req.name.strip() or f"{left.filename.rsplit('.', 1)[0]}_merged_{right.filename.rsplit('.', 1)[0]}.csv"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    session = create_session(merged, name)
    resp = _session_response(session, merged)
    resp["matched_rows"] = merged.shape[0]
    return resp


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
        # format="mixed" makes pandas infer the format row-by-row instead of
        # guessing one format from the first value and applying it to the
        # whole column. Without it, a column mixing "01-09-2026" and
        # "09/09/2026" silently turns whichever style doesn't match the
        # guessed format into missing data (NaT) — no error, just quietly
        # dropped dates. dayfirst=True still applies per-row so "01-09-2026"
        # continues to mean 1 Sept, matching DD/MM/YYYY conventions.
        result = pd.to_datetime(series, dayfirst=True, format="mixed", errors="coerce")
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


class SplitColumnRequest(BaseModel):
    column: str = Field(min_length=1)
    separator: str = Field(min_length=1, max_length=20)
    drop_original: bool = False
    new_column_prefix: str = Field(default="", max_length=60)


def _split_column_by_separator(df: pd.DataFrame, column: str, raw_sep: str,
                                drop_original: bool, prefix: str) -> tuple[pd.DataFrame, str]:
    """Split one text column into several new columns wherever `raw_sep`
    appears -- the single-column equivalent of choosing a separator at
    upload time. Accepts the same separator names/aliases (comma, tab, |, a
    custom regex like '[,;]', ...) as the upload separator field.

    The number of new columns is the *typical* (most common) field count
    across rows, not the maximum. Splitting with no cap lets a handful of
    malformed rows (e.g. one row with extra stray separators) balloon every
    row up to that row's width -- an 8-field column could silently become
    23 columns because of 2 bad rows. Capping at the typical width keeps
    well-formed rows clean; any row with *more* separators than that just
    keeps its extra text glued onto the last column instead of spawning new
    ones, and that's reported back so it's not silently lossy.
    """
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found.")
    sep = _normalize_sep(raw_sep)
    if not sep:
        raise HTTPException(400, "Please provide a separator.")

    series = df[column].astype(str)
    is_regex = len(sep) != 1

    try:
        # .str.count()/.str.findall() always treat their pattern as regex, so a
        # literal single-char separator that happens to be a regex metachar
        # (e.g. '|', '.', '$') must be escaped here even though it's passed
        # un-escaped to .str.split() below (which has a real literal mode).
        count_pat = sep if is_regex else re.escape(sep)
        field_counts = series.str.findall(count_pat).str.len() + 1
    except Exception as e:
        raise HTTPException(400, f"Could not split '{column}': {e}")

    mode = field_counts.mode()
    if mode.empty:
        raise HTTPException(400, f"The separator '{raw_sep}' never appears in '{column}' -- nothing to split.")
    expected_parts = int(mode.iloc[0])
    if expected_parts < 2:
        raise HTTPException(400, f"The separator '{raw_sep}' never appears in '{column}' -- nothing to split.")

    try:
        parts = series.str.split(sep, n=expected_parts - 1, expand=True, regex=True) if is_regex \
            else series.str.split(sep, n=expected_parts - 1, expand=True)
    except Exception as e:
        raise HTTPException(400, f"Could not split '{column}': {e}")

    n_parts = parts.shape[1]  # == expected_parts, unless every row happens to have fewer fields
    base = prefix.strip() or column
    existing = set(df.columns)
    new_names = []
    for i in range(n_parts):
        candidate = f"{base}_{i + 1}"
        while candidate in existing or candidate in new_names:
            candidate += "_"
        new_names.append(candidate)
    parts.columns = new_names
    for c in new_names:
        parts[c] = parts[c].str.strip()
    parts = parts.replace({"None": pd.NA, "": pd.NA})

    out = pd.concat([df, parts], axis=1)
    if drop_original:
        out = out.drop(columns=[column])

    too_many = int((field_counts > expected_parts).sum())
    too_few = int((field_counts < expected_parts).sum())
    note = f"Split '{column}' by '{raw_sep}' into {n_parts} new column(s): {', '.join(new_names)}."
    note += f" Original column '{column}' removed." if drop_original else f" Original column '{column}' kept."
    if too_many:
        note += (f" {too_many:,} row(s) had more '{raw_sep}' than usual -- the extra text was kept in the "
                 f"last column ('{new_names[-1]}') rather than creating more columns.")
    if too_few:
        note += f" {too_few:,} row(s) had fewer '{raw_sep}' than usual -- their trailing column(s) are blank."
    return out, note


@app.post("/api/split_column/{session_id}")
def split_column(session_id: str, req: SplitColumnRequest):
    session = _get_session_or_404(session_id)
    session.snapshot_before_change()
    session.df, log_entry = _split_column_by_separator(
        session.df, req.column, req.separator, req.drop_original, req.new_column_prefix,
    )
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


def _apply_multi_filters(df: pd.DataFrame, filters_json: str | None) -> pd.DataFrame:
    """Restricts df to rows matching every condition in filters_json — a
    JSON-encoded list of {"col": ..., "values": [...]} objects, ANDed
    together. This is the multi-column sibling of _apply_compare_filter
    (which only ever supports one column); the frontend's "Filters" panel
    uses this once the person has more than one slicer active at once.
    Returns df unchanged if no filters were supplied.
    """
    if not filters_json:
        return df
    try:
        conditions = json.loads(filters_json)
    except (TypeError, ValueError):
        raise ValueError("Could not read the filter list — try reapplying your filters.")
    if not isinstance(conditions, list):
        raise ValueError("Could not read the filter list — try reapplying your filters.")
    result = df
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        col = cond.get("col")
        values = cond.get("values")
        if not col or not values:
            continue
        if col not in df.columns:
            raise ValueError(f"Filter column '{col}' not found.")
        wanted = [str(v) for v in values]
        result = result[result[col].astype(str).isin(wanted)]
    if result.empty:
        raise ValueError("No rows match the selected filters — try unchecking one of them.")
    return result


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
                     filter_col: str | None = None, filter_values: str | None = None,
                     filters: str | None = None):
    session = _get_session_or_404(session_id)
    try:
        df = _apply_compare_filter(session.df, filter_col, filter_values)
        df = _apply_multi_filters(df, filters)

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
        if filters:
            try:
                chart["filters"] = json.loads(filters)
            except (TypeError, ValueError):
                pass
        insight = narrate.explain_chart(chart_type, x, df, y=y, group=group)
        if insight:
            chart["insight"] = insight
        session.last_visualization = chart
        return chart
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------- Simulate ----------

@app.post("/api/pivot/{session_id}")
def pivot_table(session_id: str, req: pivot_module.PivotRequest):
    """Build a pivot table from the session's current dataset (read-only)."""
    session = _get_session_or_404(session_id)
    try:
        return pivot_module.build_pivot(session.df, req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class PivotToTabRequest(pivot_module.PivotRequest):
    new_name: str = Field(default="pivot_table", max_length=120)


@app.post("/api/pivot_to_tab/{session_id}")
def pivot_to_tab(session_id: str, req: PivotToTabRequest):
    """
    Build a pivot table and turn it into a brand-new session, same pattern
    as /api/filter (Filter & Split): the summarised table becomes its own
    dataset tab -- original stays untouched -- so it can be studied further
    (visualised, filtered, cleaned, trained on) instead of only existing as
    a one-off view in the pivot panel.
    """
    session = _get_session_or_404(session_id)
    try:
        result = pivot_module.build_pivot(session.df, req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    result_df = pivot_module.pivot_rows_to_dataframe(result)
    if result_df.empty:
        raise HTTPException(400, "The pivot table has no rows to send over.")

    new_name = req.new_name.strip() or "pivot_table"
    new_session = create_session(result_df, new_name + ".csv")

    return {
        "session_id": new_session.id,
        "name": new_name,
        "shape": {"rows": len(result_df), "columns": len(result_df.columns)},
        "columns": list(result_df.columns),
        "preview": _json_safe_records(result_df.head(5)),
    }


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
    has_forecast     = bool(session.forecast_results.get("latest"))

    if has_training or has_predictions:
        stage = "Model Training & Prediction"
    elif has_unsupervised:
        stage = "Unsupervised Learning"
    elif has_forecast:
        stage = "Trend & Forecast"
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

    forecast_result = session.forecast_results.get("latest")
    if forecast_result:
        f = forecast_result
        lines.extend(["", "## Trend & Forecast", ""])
        degree_label = "Quadratic" if f.get("degree") == 2 else "Linear"
        period_label = "numeric points" if f.get("period") == "numeric" else f"{f.get('period', 'year')}s"
        metric_label = f"{f.get('agg', 'count')} of {f.get('metric_col')}" if f.get("metric_col") else "row count"
        r2 = f.get("r2")
        lines.append(f"**{degree_label} trend** over {f.get('n_points', 0)} {period_label} — column: `{f.get('date_col', '')}`, metric: {metric_label}")
        lines.append(f"- R² = {_fmt_report_value(r2) if r2 is not None else 'n/a'}")
        lines.append(f"- {f.get('equation', '')}")
        for w in (f.get("warnings") or []):
            lines.append(f"- ⚠ {w}")
        hist = f.get("historical", {})
        future = f.get("future", {})
        if hist.get("labels") or future.get("labels"):
            lines.append("")
            lines.append("| Period | Actual | Forecast |")
            lines.append("| --- | --- | --- |")
            for label, value in zip(hist.get("labels", []), hist.get("values", [])):
                lines.append(f"| {label} | {_fmt_report_value(value)} |  |")
            for label, value in zip(future.get("labels", []), future.get("values", [])):
                lines.append(f"| {label} |  | {_fmt_report_value(value)} |")
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


@app.get("/api/preview/{session_id}")
def get_preview_range(
    session_id: str,
    which: Literal["current", "original"] = "current",
    start: int = 1,
    end: int | None = None,
    n: int | None = None,
):
    """Preview any slice of rows -- like df.head(n), but with a start row and
    either a count (n) or an explicit end row, e.g. rows 20 through 30.
    `start`/`end` are 1-indexed and inclusive, matching the '# Row' column
    shown in the UI. `which` picks the current (possibly cleaned) data or the
    frozen original upload."""
    session = _get_session_or_404(session_id)
    df = session.df if which == "current" else session.original_df
    total = len(df)

    start = max(1, start)
    if end is not None:
        if end < start:
            raise HTTPException(400, "The end row must be greater than or equal to the start row.")
        count = end - start + 1
    else:
        count = max(1, n or 15)
    count = min(count, 5000)  # sane cap so a typo'd range can't return the whole dataset as one payload

    start_idx = start - 1
    records = eda.safe_preview(df, n=count, start=start_idx) if start_idx < total else []
    actual_end = start_idx + len(records)  # last row actually returned (1-indexed, inclusive)

    return {
        "preview": records,
        "start": start,
        "end": actual_end,
        "total_rows": total,
        "returned": len(records),
    }


def _write_xlsx(df: pd.DataFrame) -> io.BytesIO:
    """Write a DataFrame to .xlsx by hand with xlsxwriter's constant_memory
    mode, which streams rows straight to disk instead of buffering the whole
    workbook as Python objects (what pandas.to_excel()/openpyxl do) -- this
    keeps memory flat no matter how large the export is.

    We can't use pandas.to_excel() with this mode: pandas writes cells
    through xlsxwriter in a way that breaks constant_memory's "writes must
    stay in strict left-to-right row order" requirement as soon as a row has
    more than one cell format (e.g. any text column next to any numeric
    column) -- every column after the first silently comes back blank for
    every row. Writing cell-by-cell here, in guaranteed row order, gets both
    correctness and the memory benefit, for small and large exports alike.
    """
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"constant_memory": True})
    ws = wb.add_worksheet("Sheet1")
    date_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})
    datetime_fmt = wb.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})

    columns = list(df.columns)
    ws.write_row(0, 0, [str(c) for c in columns])

    is_datetime_col = [pd.api.types.is_datetime64_any_dtype(df[c]) for c in columns]
    col_has_time = []
    for i, c in enumerate(columns):
        if not is_datetime_col[i]:
            col_has_time.append(False)
            continue
        s = df[c]
        col_has_time.append(bool(((s.dt.hour != 0) | (s.dt.minute != 0) | (s.dt.second != 0)).any()))

    for row_idx, row in enumerate(df.itertuples(index=False, name=None), start=1):
        for col_idx, v in enumerate(row):
            if v is None or v is pd.NaT or (isinstance(v, float) and pd.isna(v)):
                ws.write_blank(row_idx, col_idx, None)
            elif is_datetime_col[col_idx]:
                dt = v.to_pydatetime() if hasattr(v, "to_pydatetime") else v
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)  # xlsxwriter dates must be timezone-naive
                ws.write_datetime(row_idx, col_idx, dt, datetime_fmt if col_has_time[col_idx] else date_fmt)
            elif isinstance(v, (bool, np.bool_)):
                ws.write_boolean(row_idx, col_idx, bool(v))
            elif isinstance(v, (int, np.integer)):
                ws.write_number(row_idx, col_idx, int(v))
            elif isinstance(v, (float, np.floating)):
                ws.write_number(row_idx, col_idx, float(v))
            else:
                ws.write_string(row_idx, col_idx, str(v))

    wb.close()
    buf.seek(0)
    return buf


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
        buf = _write_xlsx(df)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={name}_autods.xlsx", "Cache-Control": "no-store"},
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
        headers={"Content-Disposition": f"attachment; filename={name}_autods.csv", "Cache-Control": "no-store"},
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
        headers={"Content-Disposition": f"attachment; filename={name}_autods_work_report.md", "Cache-Control": "no-store"},
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
        headers={"Content-Disposition": f"attachment; filename={name}_autods_report.html", "Cache-Control": "no-store"},
    )


@app.post("/api/download_html/{session_id}")
def download_html_with_custom_charts(session_id: str, req: ReportChartsRequest):
    session = _get_session_or_404(session_id)
    if req.charts:
        # Remember the real dashboard so a later request that can't carry
        # its own charts (e.g. a reloaded/reopened tab re-issuing this as a
        # plain GET) can still fall back to it instead of generic
        # auto-suggestions — see _build_html_report's fallback chain.
        session.staged_charts = req.charts
    report = _build_html_report(session, extra_charts=req.charts).encode("utf-8")
    buf = io.BytesIO(report)
    name = session.filename.rsplit(".", 1)[0]
    return StreamingResponse(
        buf,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={name}_autods_report.html", "Cache-Control": "no-store"},
    )


def _parse_form_charts(charts_json: str) -> list:
    try:
        parsed = json.loads(charts_json) if charts_json else []
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


@app.post("/api/download_html_form/{session_id}")
def download_html_form(session_id: str, charts_json: str = Form("[]")):
    """
    Same report as download_html_with_custom_charts, but bound to a real
    HTML <form> submission (application/x-www-form-urlencoded) instead of a
    JSON body.

    The frontend used to open a blank tab, POST the chart history to
    /api/stage_charts as JSON, then redirect that tab to the plain GET
    download once the request resolved. That works most of the time, but it
    depends on the browser's "user activation" window from the original
    click still being valid when the async fetch resolves — if that fetch
    is slow, or the browser is strict about it (mobile Safari especially),
    the activation expires and the browser silently refuses the later
    redirect, leaving the tab stuck on about:blank forever with no error at
    all. A real <form method="post" target="_blank"> submit is a genuine
    top-level navigation triggered synchronously by the click itself — the
    same reliable mechanism the working CSV/Excel/.md downloads already
    use — so there's no async gap for a browser to get suspicious about,
    and no separate staging round-trip needed either.
    """
    session = _get_session_or_404(session_id)
    parsed_charts = _parse_form_charts(charts_json)
    if parsed_charts:
        # Same reasoning as download_html_with_custom_charts above: persist
        # the real dashboard so a later GET on this same URL (reload,
        # history reopen, pasted link) has something real to fall back to.
        session.staged_charts = parsed_charts
    report = _build_html_report(session, extra_charts=parsed_charts).encode("utf-8")
    buf = io.BytesIO(report)
    name = session.filename.rsplit(".", 1)[0]
    return StreamingResponse(
        buf,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={name}_autods_report.html", "Cache-Control": "no-store"},
    )


@app.get("/api/download_html_form/{session_id}")
def download_html_form_get(session_id: str, charts_json: str = "[]"):
    """
    GET counterpart to download_html_form, at the exact same URL. The app
    itself always POSTs here via the real form (see the docstring above),
    but this same URL can also end up being opened directly — e.g. someone
    reloads that tab, retypes/pastes the URL, or reopens it from browser
    history — and a browser always issues a plain GET in those cases no
    matter how the page first got there. Without this, that would 405 with
    a bare "Method Not Allowed" and no way to recover except going back to
    the app and clicking the button again. Answering GET too means the
    report always comes back (just without any custom charts staged on this
    particular request, since those only travel with the original POST).
    """
    session = _get_session_or_404(session_id)
    report = _build_html_report(session, extra_charts=_parse_form_charts(charts_json)).encode("utf-8")
    buf = io.BytesIO(report)
    name = session.filename.rsplit(".", 1)[0]
    return StreamingResponse(
        buf,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={name}_autods_report.html", "Cache-Control": "no-store"},
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

# Hard ceiling on how long a single training run is allowed to run before it
# is killed automatically -- independent of whether anyone is watching the
# page or clicks Cancel. Without this, a run that gets stuck (e.g. thrashing
# on a slow/overloaded machine, or a pathological column) can sit there for
# hours or days with nothing to stop it, since the old behaviour only ever
# stopped training in response to an explicit user click. Configurable via
# env var so it can be tuned per deployment without editing code.
TRAIN_TIMEOUT_SECONDS = int(os.environ.get("AUTODS_TRAIN_TIMEOUT_SECONDS", 20 * 60))


def _watchdog_timeout(session_id: str, pid: int):
    """Runs on a background timer thread. If the job for session_id is still
    the same process (matched by pid, in case a new run already replaced it)
    and it's still alive once TRAIN_TIMEOUT_SECONDS has elapsed, kill it and
    flag it as a timeout (not a user cancel) so the status endpoint can report
    an honest, specific message instead of a misleading "cancelled"."""
    job = TRAIN_JOBS.get(session_id)
    if job is None or job["process"].pid != pid:
        return
    if job["process"].is_alive():
        job["timed_out"] = True
        try:
            job["process"].terminate()
        except Exception:
            pass


def _train_worker(df, target: str, use_pca: bool, progress_list, result_queue):
    """Runs in a child process. Talks back to the parent only via progress_list/result_queue."""
    import time
    import traceback

    worker_start = time.time()

    def _report(msg: str):
        elapsed = time.time() - worker_start
        # Prefix every message with elapsed time so if the process is killed
        # we can see exactly where it got to and how long it had been running.
        progress_list.append(f"[{elapsed:.0f}s] {msg}")

    # Log memory at startup so slow-start OOM issues are visible.
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        mem_mb = proc.memory_info().rss / 1_048_576
        vm = psutil.virtual_memory()
        _report(f"Worker started — process memory: {mem_mb:.0f} MB | system free: {vm.available // 1_048_576} MB")
    except Exception:
        _report("Worker started.")

    try:
        problem_type, leaderboard, fitted, best_name, label_encoder, best_importance = automl.train_all(
            df, target, use_pca=use_pca, progress_callback=_report
        )

        elapsed_total = time.time() - worker_start
        _report(f"All models trained in {elapsed_total:.1f}s — packaging results...")

        result_queue.put({
            "status": "done",
            "problem_type": problem_type,
            "leaderboard": leaderboard,
            "fitted": fitted,
            "best_name": best_name,
            "label_encoder": label_encoder,
            "feature_importance": best_importance,
        })
    except Exception as e:
        elapsed_total = time.time() - worker_start
        tb = traceback.format_exc()
        result_queue.put({
            "status": "error",
            "error": f"{e}\n\nTraining had been running for {elapsed_total:.1f}s.\n\nFull traceback:\n{tb}",
        })


def _cleanup_train_job(session_id: str):
    job = TRAIN_JOBS.pop(session_id, None)
    if job is None:
        return
    watchdog = job.get("watchdog")
    if watchdog is not None:
        watchdog.cancel()
    if job["process"].is_alive():
        job["process"].terminate()
        job["process"].join(timeout=2)
        if job["process"].is_alive():
            job["process"].kill()
    job["manager"].shutdown()


def _best_model_feature_importance(models: dict, best_model_name: str | None) -> list:
    """Feature importance for the leaderboard winner. Takes only the fitted
    pipeline (feature names/importances come from what's already fitted inside
    it) -- no need to hand over a copy of the training dataframe, which used to
    be built fresh at every save/status check and could be sizeable on a large
    dataset."""
    if best_model_name and best_model_name in models:
        return automl.feature_importance(models[best_model_name])
    return []


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
            # Reuse the importance already cached for the CURRENT (about-to-be-
            # replaced) target rather than recomputing from the bare fitted
            # pipeline with no data -- session.feature_importance was filled in
            # when this model finished training and is exactly what belongs to
            # session.models/session.best_model_name at this point.
            "feature_importance": session.feature_importance,
        }

    # Starting a new run always replaces any previous one for this session.
    _cleanup_train_job(session_id)

    # HistGradientBoosting (and other OpenMP-parallel sklearn code) can
    # deadlock permanently -- not just run slow, but hang forever -- when the
    # worker process is created with the platform-default "fork" start
    # method, because it clones the parent's already-initialized OpenMP
    # thread pool along with it, and that state doesn't survive the fork
    # correctly. "spawn" starts a clean interpreter with no inherited thread
    # state, which avoids this entirely. This is a little slower to start
    # (a few hundred ms) but that's a rounding error next to a training run
    # that would otherwise hang indefinitely.
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress = manager.list(["Starting training...", "Preparing data and features..."])
    result_queue = manager.Queue()

    process = ctx.Process(
        target=_train_worker,
        args=(session.df.copy(), req.target, req.use_pca, progress, result_queue),
        daemon=True,
    )
    process.start()

    watchdog = threading.Timer(TRAIN_TIMEOUT_SECONDS, _watchdog_timeout, args=(session_id, process.pid))
    watchdog.daemon = True
    watchdog.start()

    TRAIN_JOBS[session_id] = {
        "process": process,
        "manager": manager,
        "progress": progress,
        "result_queue": result_queue,
        "target": req.target,
        "cancel_requested": False,
        "timed_out": False,
        "n_rows": len(session.df),
        "started_at": time.time(),
        "watchdog": watchdog,
    }

    return {"status": "started"}


@app.get("/api/train/{session_id}/diag")
def train_diag(session_id: str):
    """Diagnostic endpoint — returns full progress log, elapsed time, memory
    usage, and process state. Open in browser while training is running to see
    exactly which model was active and how much memory is being used."""
    import time, psutil

    job = TRAIN_JOBS.get(session_id)
    if job is None:
        return {"status": "idle", "messages": [], "elapsed_s": 0}

    messages = list(job["progress"])
    alive = job["process"].is_alive()
    pid = job["process"].pid

    mem_info = {}
    if pid:
        try:
            proc = psutil.Process(pid)
            rss_mb = proc.memory_info().rss / 1_048_576
            vm = psutil.virtual_memory()
            mem_info = {
                "worker_rss_mb": round(rss_mb, 1),
                "system_free_mb": round(vm.available / 1_048_576, 1),
                "system_used_pct": vm.percent,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            mem_info = {"note": "process ended before memory could be read"}

    started_at = job.get("started_at", 0)
    elapsed = round(time.time() - started_at, 1) if started_at else None
    n_rows = job.get("n_rows")

    return {
        "status": "running" if alive else "exited",
        "pid": pid,
        "elapsed_s": elapsed,
        "n_rows": n_rows,
        "messages": messages,
        **mem_info,
    }


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

    exitcode = job["process"].exitcode
    was_cancelled = job.get("cancel_requested", False)
    timed_out = job.get("timed_out", False)
    started_at = job.get("started_at")
    _cleanup_train_job(session_id)

    if result is None:
        if timed_out:
            minutes = TRAIN_TIMEOUT_SECONDS // 60
            return {
                "status": "error",
                "messages": messages,
                "error": (
                    f"Training was automatically stopped after running for {minutes} minutes "
                    "without finishing. This is a safety limit so a run can never be left going "
                    "indefinitely (e.g. overnight or for days) even if nobody is watching. "
                    "Try again with 'Use PCA for training' enabled to reduce dimensionality, "
                    "remove very high-cardinality text/ID columns, or increase "
                    "AUTODS_TRAIN_TIMEOUT_SECONDS if this dataset genuinely needs more time."
                ),
            }
        if was_cancelled:
            return {"status": "cancelled", "messages": messages}
        # The process died without putting anything on the result queue and the
        # user never clicked Cancel -- it was killed (most likely by the OS, e.g.
        # an out-of-memory kill) or crashed natively (a segfault deep inside a
        # C/C++ extension). Either way this is NOT a user cancellation, and
        # calling it "cancelled" was actively misleading, especially on large
        # datasets where memory pressure is the most likely cause. Report it as
        # an error with an honest, specific explanation instead.
        if exitcode is not None and exitcode < 0:
            n_rows = job.get("n_rows")
            large_hint = (
                f" This dataset has {n_rows:,} rows, which increases memory pressure during training."
                if n_rows and n_rows > automl.LARGE_DATASET_THRESHOLD else ""
            )
            fast_hint = (
                f" Try again with 'Use PCA for training' enabled, or remove very high-cardinality "
                f"categorical/text/ID columns to speed up training."
                if n_rows and n_rows > automl.FAST_MODELS_THRESHOLD else ""
            )
            if exitcode == -9:
                error = (
                    "Training was stopped by the system, most likely because it ran out of memory."
                    + large_hint
                    + fast_hint
                    + " Try again, or if this keeps happening, remove very high-cardinality text columns "
                      "(e.g. free-text titles or addresses) before training."
                )
            elif exitcode == -15:
                error = (
                    "Training was stopped by a SIGTERM signal (signal 15). "
                    "The most common cause is uvicorn's hot-reload: if you're running the server "
                    "with 'python backend/run.py' (no --dev flag), this should not happen. "
                    "If you are using --dev mode, any file change inside backend/ will restart "
                    "the server mid-training. Check the diagnostic log at "
                    "/api/train/<session_id>/diag for the full progress before it was killed."
                )
            else:
                error = (
                    f"Training crashed unexpectedly (signal {-exitcode})." + large_hint + fast_hint
                    + " Try again; if it keeps happening, please report this."
                )
        else:
            error = "Training exited unexpectedly without reporting a result. Please try again."
        return {"status": "error", "messages": messages, "error": error}

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

    # Computed inside the worker (train_all), where the held-out X_test/y_test
    # split was still available for the permutation-importance fallback --
    # recomputing it out here from just the fitted pipeline would lose that
    # data and always come back empty for models like HistGradientBoosting
    # that have no feature_importances_/coef_ of their own.
    importance = result.get("feature_importance") or []
    session.feature_importance = importance  # cache so report + auto-save carry it

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
    job["cancel_requested"] = True
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
        # Older saved runs (from before this field existed) fall back to [] --
        # the frontend already treats an empty list as "not available".
        "feature_importance": run.get("feature_importance", []),
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
        # Same reasoning as the auto-save above: session.feature_importance is
        # already the correct importance for session.models/best_model_name
        # (computed with real held-out data back when training finished), so
        # reuse it instead of recomputing from the bare pipeline with none.
        "feature_importance": session.feature_importance,
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


def _resolve_trained_model(session, run_id: str | None, model_name: str | None):
    """Look up which trained model to predict with — either the session's
    current live training or one of its saved runs — shared by the single-row
    and batch predict endpoints so they can never disagree on how a run_id
    or model_name resolves.

    Returns (pipe, model_name, problem_type, target, label_encoder,
    feature_columns, source_name).
    """
    if run_id:
        if run_id not in session.saved_runs:
            raise HTTPException(404, "Saved run not found.")
        run = session.saved_runs[run_id]
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

    resolved_name = model_name or default_model_name
    if resolved_name not in models:
        raise HTTPException(400, f"Model '{resolved_name}' not found.")

    return (
        models[resolved_name], resolved_name, problem_type, target,
        label_encoder, feature_columns, source_name,
    )


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
    pipe, model_name, problem_type, target, label_encoder, feature_columns, source_name = (
        _resolve_trained_model(session, req.run_id, req.model_name)
    )

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


# Column-name patterns that almost always mean "row identifier" rather than a
# feature — used to auto-pick the id column for the Kaggle-style two-column
# submission CSV in batch predict, without making the user hunt for it.
_ID_COLUMN_RE = re.compile(r"(?:^id$|_id$|^id_|id$)", re.IGNORECASE)


def _guess_id_column(columns: list[str]) -> str | None:
    for col in columns:
        if _ID_COLUMN_RE.search(col):
            return col
    return None


class BatchPredictRequest(BaseModel):
    target_session_id: str   # the dataset to score (e.g. test.csv's session)
    run_id: str | None = None
    model_name: str | None = None


@app.post("/api/predict_batch/{model_session_id}")
def predict_batch(model_session_id: str, req: BatchPredictRequest):
    """Score every row of a *different* dataset (req.target_session_id) with
    a model trained on model_session_id, and return the results as CSV —
    the batch counterpart to /api/predict, which only ever takes up to 100
    hand-typed rows against the model's own session. This is what makes a
    train.csv / test.csv Kaggle-style split actually usable: train on one
    dataset tab, then score another tab's rows with that model without
    retyping anything.
    """
    model_session = _get_session_or_404(model_session_id)
    target_session = _get_session_or_404(req.target_session_id)

    pipe, model_name, problem_type, target, label_encoder, feature_columns, source_name = (
        _resolve_trained_model(model_session, req.run_id, req.model_name)
    )

    input_df = target_session.df.copy()
    if input_df.empty:
        raise HTTPException(400, "The dataset to score has no rows.")

    missing_cols = [c for c in feature_columns if c not in input_df.columns]
    scoring_df = input_df.copy()
    for c in missing_cols:
        scoring_df[c] = None
    scoring_df = scoring_df[feature_columns]

    preds = pipe.predict(scoring_df)
    if problem_type == "classification" and label_encoder is not None:
        preds = label_encoder.inverse_transform(preds.astype(int))

    # "Full" output: every original column from the target dataset plus the
    # prediction, in case the user wants the extra context (e.g. to sanity-
    # check predictions against Age/Sex/Fare visually).
    full_df = input_df.copy()
    full_df[target] = preds

    id_column = _guess_id_column(list(input_df.columns))
    submission_csv = None
    if id_column:
        submission_df = full_df[[id_column, target]]
        submission_csv = submission_df.to_csv(index=False)

    preview_df = full_df.head(10)
    preview = [
        {k: _jsonable_value(v) for k, v in row.items()}
        for row in preview_df.to_dict(orient="records")
    ]

    return {
        "ok": True,
        "target": target,
        "problem_type": problem_type,
        "model_name": model_name,
        "source_name": source_name,
        "rows_predicted": int(len(full_df)),
        "missing_feature_columns": missing_cols,
        "id_column": id_column,
        "full_csv": full_df.to_csv(index=False),
        "submission_csv": submission_csv,
        "preview": preview,
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


def _imported_model_summary(model_id: str, entry: dict) -> dict:
    return {
        "id": model_id,
        "model_label": entry["model_label"],
        "problem_type": entry["problem_type"],
        "feature_columns": entry["feature_columns"],
        "target_name": entry["target_name"],
        "has_label_classes": entry.get("label_classes") is not None,
        "original_filename": entry["original_filename"],
        "imported_at": entry["imported_at"],
    }


def _imported_models_summary(session) -> list[dict]:
    return [_imported_model_summary(mid, e) for mid, e in session.imported_models.items()]


@app.post("/api/import_model/{session_id}")
async def import_model(
    session_id: str,
    file: UploadFile = File(...),
    problem_type: str | None = Form(None),
    feature_columns: str | None = Form(None),  # JSON array of column names, e.g. '["age","income"]'
    target_name: str = Form(""),
    label_classes: str | None = Form(None),    # JSON array of class labels, in encoded order
    model_label: str = Form(""),
):
    """Import a model trained OUTSIDE AutoDS (e.g. by someone else, in their
    own notebook) so it can be used for predictions here. Unlike a trained-
    in-AutoDS model, this is read-only: it never appears on the leaderboard,
    can't be retrained, and nothing about it is assumed beyond "it has a
    .predict() method" -- see _extract_model_pipeline/_safe_unpickle_model.

    problem_type and feature_columns are the two things AutoDS has no way
    to know about a model it didn't train, so they're required UNLESS the
    file is itself an AutoDS download_model bundle (which already carries
    them) or the pipeline recorded its own feature_names_in_ (scikit-learn
    does this automatically when fit on a DataFrame).
    """
    session = _get_session_or_404(session_id)

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"Model file is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    try:
        obj = _safe_unpickle_model(contents)
    except UnsafeModelFileError as e:
        raise HTTPException(400, f"This file can't be imported: {e}. Only models built with scikit-learn, "
                                  f"numpy, scipy, pandas, xgboost, lightgbm or catboost are supported.")
    except ModuleNotFoundError as e:
        raise HTTPException(400, f"This model needs a package that isn't installed here ({e}).")
    except Exception as e:
        raise HTTPException(400, f"Couldn't read this as a model file: {e}")

    pipeline = _extract_model_pipeline(obj)
    if pipeline is None:
        raise HTTPException(400, "This file doesn't look like a trained model — no predict() method found.")

    bundle = obj if isinstance(obj, dict) else {}

    resolved_problem_type = problem_type or bundle.get("problem_type")
    if resolved_problem_type not in ("classification", "regression"):
        raise HTTPException(400, "problem_type must be 'classification' or 'regression' "
                                  "(this file didn't already carry one, so it needs to be specified).")

    resolved_feature_columns = None
    if feature_columns:
        try:
            parsed = json.loads(feature_columns)
            if not isinstance(parsed, list) or not parsed or not all(isinstance(c, str) for c in parsed):
                raise ValueError
            resolved_feature_columns = parsed
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(400, "feature_columns must be a JSON array of column names.")
    if resolved_feature_columns is None:
        bundle_cols = bundle.get("feature_columns")
        if isinstance(bundle_cols, list) and bundle_cols:
            resolved_feature_columns = list(bundle_cols)
    if resolved_feature_columns is None:
        names_in = getattr(pipeline, "feature_names_in_", None)
        if names_in is not None:
            resolved_feature_columns = list(names_in)
    if not resolved_feature_columns:
        raise HTTPException(400, "feature_columns is required — this model didn't record its own feature "
                                  "names, so list the columns it expects to see, in the order it expects them.")

    resolved_label_classes = None
    if label_classes:
        try:
            parsed = json.loads(label_classes)
            if not isinstance(parsed, list):
                raise ValueError
            resolved_label_classes = parsed
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(400, "label_classes must be a JSON array.")
    if resolved_label_classes is None:
        bundle_encoder = bundle.get("label_encoder")
        classes = getattr(bundle_encoder, "classes_", None)
        if classes is not None:
            resolved_label_classes = [_jsonable_value(c) for c in classes]

    resolved_target_name = target_name.strip() or bundle.get("target") or "prediction"

    model_id = str(uuid.uuid4())
    session.imported_models[model_id] = {
        "pipeline": pipeline,
        "problem_type": resolved_problem_type,
        "feature_columns": resolved_feature_columns,
        "target_name": resolved_target_name,
        "label_classes": resolved_label_classes,
        "model_label": model_label.strip() or (file.filename or "Imported model"),
        "original_filename": file.filename or "model.pkl",
        "imported_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return {"imported_models": _imported_models_summary(session)}


@app.get("/api/imported_models/{session_id}")
def list_imported_models(session_id: str):
    session = _get_session_or_404(session_id)
    return {"imported_models": _imported_models_summary(session)}


@app.delete("/api/imported_models/{session_id}/{model_id}")
def delete_imported_model(session_id: str, model_id: str):
    session = _get_session_or_404(session_id)
    if model_id not in session.imported_models:
        raise HTTPException(404, "Imported model not found.")
    del session.imported_models[model_id]
    return {"imported_models": _imported_models_summary(session)}


class ImportedModelPredictRequest(BaseModel):
    model_id: str
    target_session_id: str | None = None  # defaults to scoring this session's own data


@app.post("/api/imported_model_predict/{session_id}")
def imported_model_predict(session_id: str, req: ImportedModelPredictRequest):
    """Score a dataset with an imported model — the imported-model
    counterpart to /api/predict_batch, same response shape (full_csv,
    submission_csv when an id column is detectable, preview rows) so the
    frontend can reuse its existing batch-predict results UI for both."""
    session = _get_session_or_404(session_id)
    if req.model_id not in session.imported_models:
        raise HTTPException(404, "Imported model not found.")
    entry = session.imported_models[req.model_id]

    target_session = _get_session_or_404(req.target_session_id) if req.target_session_id else session
    input_df = target_session.df.copy()
    if input_df.empty:
        raise HTTPException(400, "The dataset to score has no rows.")

    feature_columns = entry["feature_columns"]
    missing_cols = [c for c in feature_columns if c not in input_df.columns]
    scoring_df = input_df.copy()
    for c in missing_cols:
        scoring_df[c] = None
    scoring_df = scoring_df[feature_columns]

    try:
        preds = entry["pipeline"].predict(scoring_df)
    except Exception as e:
        # An external model has no AutoDS-built "prep" step guaranteeing
        # clean input, so a bad/missing column is a normal, expected
        # failure here -- surface it plainly instead of a raw 500.
        raise HTTPException(400, f"This model couldn't score the data: {e}")

    label_classes = entry.get("label_classes")
    if entry["problem_type"] == "classification" and label_classes:
        decoded = []
        for p in preds:
            try:
                idx = int(p)
            except (TypeError, ValueError):
                decoded = None
                break
            decoded.append(label_classes[idx] if 0 <= idx < len(label_classes) else p)
        if decoded is not None:
            preds = decoded

    target_name = entry["target_name"]
    full_df = input_df.copy()
    full_df[target_name] = [_jsonable_value(p) for p in preds]

    id_column = _guess_id_column(list(input_df.columns))
    submission_csv = None
    if id_column:
        submission_csv = full_df[[id_column, target_name]].to_csv(index=False)

    preview_df = full_df.head(10)
    preview = [
        {k: _jsonable_value(v) for k, v in row.items()}
        for row in preview_df.to_dict(orient="records")
    ]

    return {
        "ok": True,
        "model_label": entry["model_label"],
        "problem_type": entry["problem_type"],
        "target_name": target_name,
        "rows_predicted": int(len(full_df)),
        "missing_feature_columns": missing_cols,
        "id_column": id_column,
        "full_csv": full_df.to_csv(index=False),
        "submission_csv": submission_csv,
        "preview": preview,
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


# ---------- Recommender Systems ----------
# Works on any dataset: Netflix-style user/item ratings, implicit
# interactions (clicks/purchases/watches), or a single item-feature table
# (content-based "more like this"). See backend/recommender.py.

@app.get("/api/recommender/detect/{session_id}")
def recommender_detect(session_id: str):
    """Inspect the current dataset and suggest the best recommender mode + columns."""
    session = _get_session_or_404(session_id)
    return recommender.detect_recommender_setup(session.df)


@app.post("/api/recommender/fit/{session_id}")
def recommender_fit(session_id: str, req: dict):
    """
    Fit a recommender model and cache it in-memory on the session.
    req:
      mode: "collaborative" | "implicit" | "content"
      collaborative/implicit: user_col, item_col, rating_col (optional), n_components
      content: item_col, feature_cols (list), text_cols (list, optional), list_cols (list, optional)
    Returns {model_id, ...summary stats...} -- model_id is used by the
    recommend/similar endpoints below.
    """
    session = _get_session_or_404(session_id)
    mode = req.get("mode")
    try:
        if mode in ("collaborative", "implicit"):
            user_col = req.get("user_col")
            item_col = req.get("item_col")
            rating_col = req.get("rating_col") if mode == "collaborative" else None
            if not user_col or not item_col:
                raise HTTPException(400, "user_col and item_col are required.")
            model = recommender.fit_collaborative(
                session.df, user_col, item_col, rating_col,
                n_components=req.get("n_components", 20),
            )
        elif mode == "content":
            item_col = req.get("item_col")
            feature_cols = req.get("feature_cols") or []
            if not item_col or not feature_cols:
                raise HTTPException(400, "item_col and feature_cols are required.")
            model = recommender.fit_content(
                session.df, item_col, feature_cols,
                text_cols=req.get("text_cols"), list_cols=req.get("list_cols"),
            )
        else:
            raise HTTPException(400, f"Unknown recommender mode: {mode}")
    except ValueError as e:
        raise HTTPException(400, str(e))

    model_id = str(uuid.uuid4())
    session.recommender_models[model_id] = model
    summary = {k: v for k, v in model.items() if not k.startswith("_")}
    summary["model_id"] = model_id
    session.recommender_results[model_id] = summary
    return summary


@app.get("/api/recommender/models/{session_id}")
def recommender_list_models(session_id: str):
    """List previously fitted recommender models for this session."""
    session = _get_session_or_404(session_id)
    return {"models": list(session.recommender_results.values())}


@app.get("/api/recommender/recommend/{session_id}/{model_id}")
def recommender_recommend(session_id: str, model_id: str, user_id: str, top_n: int = 10, method: str = "hybrid"):
    """Recommend unseen items for a user (collaborative/implicit models only)."""
    session = _get_session_or_404(session_id)
    model = session.recommender_models.get(model_id)
    if model is None:
        raise HTTPException(404, "Recommender model not found. Fit one first with /api/recommender/fit.")
    if model["mode"] not in ("collaborative", "implicit"):
        raise HTTPException(400, "This endpoint needs a collaborative/implicit model; use /api/recommender/similar for a content model.")
    try:
        return recommender.recommend_for_user(model, user_id, top_n=top_n, method=method)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/recommender/similar/{session_id}/{model_id}")
def recommender_similar(session_id: str, model_id: str, item: str, top_n: int = 10):
    """'More like this' -- similar items for a content model, or item-item similarity for a collaborative model."""
    session = _get_session_or_404(session_id)
    model = session.recommender_models.get(model_id)
    if model is None:
        raise HTTPException(404, "Recommender model not found. Fit one first with /api/recommender/fit.")
    try:
        if model["mode"] == "content":
            return recommender.recommend_similar(model, item, top_n=top_n)
        else:
            return recommender.similar_items_collaborative(model, item, top_n=top_n)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/recommender/evaluate/{session_id}")
def recommender_evaluate(session_id: str, req: dict):
    """Quick RMSE/MAE train/test sanity check for a collaborative model (needs a rating_col)."""
    session = _get_session_or_404(session_id)
    user_col, item_col, rating_col = req.get("user_col"), req.get("item_col"), req.get("rating_col")
    if not user_col or not item_col or not rating_col:
        raise HTTPException(400, "user_col, item_col, and rating_col are required.")
    try:
        return recommender.evaluate_collaborative(
            session.df, user_col, item_col, rating_col,
            test_size=req.get("test_size", 0.2),
            n_components=req.get("n_components", 20),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------- Trend & Forecast ----------
# "Content additions per year, projected N years forward" style workflow:
# aggregate a date/year column into periods, fit a linear/quadratic trend,
# extrapolate, and return chart-ready historical/trend/future series.
# Stateless (nothing cached on the session) since each run is cheap.

@app.get("/api/forecast/detect/{session_id}")
def forecast_detect(session_id: str):
    """Suggest a date/year column and metric to trend & forecast."""
    session = _get_session_or_404(session_id)
    return forecast.detect_forecast_setup(session.df)


@app.post("/api/forecast/run/{session_id}")
def forecast_run(session_id: str, req: dict):
    """
    req:
      date_col: required -- a date column, an already-numeric year column,
                or (with period "numeric") any numeric column at all
      period: "year" | "quarter" | "month" | "numeric" (default "year")
      metric_col: optional numeric column to aggregate (omit to count rows)
      agg: "count" | "sum" | "mean" | "median" | "min" | "max" (default "count")
      degree: 1 (linear) or 2 (quadratic trend), default 1
      periods_ahead: how many future periods to project, default 5 --
                     ignored when future_values is given
      future_values: optional list of specific numbers to predict (years,
                     for period "year"; raw numbers for "numeric") instead
                     of "N periods right after the last one" -- an
                     arbitrary range or a non-consecutive set of inputs
    """
    session = _get_session_or_404(session_id)
    date_col = req.get("date_col")
    if not date_col:
        raise HTTPException(400, "date_col is required.")
    try:
        result = forecast.run_forecast(
            session.df,
            date_col=date_col,
            period=req.get("period", "year"),
            metric_col=req.get("metric_col") or None,
            agg=req.get("agg", "count"),
            degree=req.get("degree", 1),
            periods_ahead=req.get("periods_ahead", 5),
            future_values=req.get("future_values") or None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Persist the latest run (plain JSON-safe numbers/strings -- no fitted
    # model object) so it survives tab switches and server restarts, and
    # shows up in the HTML report / notebook export, like every other
    # analysis tool's results already do. See store.py's forecast_results.
    session.forecast_results["latest"] = result
    session.save_meta_to_disk()
    return result


# ---------- Download Jupyter Notebook ----------

def _write_cleaning_engine_bundle(zf) -> None:
    """Bundle the app's own cleaning engine (clean.py + clean_chat.py and
    their two small dependencies, nlp.py and assistant.py) into the zip as
    flat, dependency-free files under autods_cleaning_tools/. The exported
    notebook's cleaning cells import clean_dataframe()/run_command() from
    these exact files instead of a hand-written re-implementation of the
    same logic, so the notebook's cleaning result is guaranteed to match
    what the app actually did instead of risking the two drifting apart."""
    import inspect

    for mod, fname in (
        (nlp, "nlp.py"),
        (assistant, "assistant.py"),
        (clean_module, "clean.py"),
        (clean_chat, "clean_chat.py"),
    ):
        src = inspect.getsource(mod)
        # These four files are normally part of the `backend` package and use
        # relative imports between each other; flatten those to plain
        # top-level imports so the files work standalone, sitting next to the
        # notebook with no package structure around them.
        src = src.replace("from . import nlp", "import nlp")
        src = src.replace("from .assistant import", "from assistant import")
        zf.writestr(f"autods_cleaning_tools/{fname}", src)


def _build_notebook_bundle_zip(session, charts: list | None) -> tuple[io.BytesIO, str]:
    """Zip the notebook together with the plain CSV file(s) it reads from
    (original_data.csv always, cleaned_data.csv too if cleaning changed
    anything) so the download is immediately runnable after unzipping --
    no separate "connect the dataset" step, and no base64-bloated .ipynb,
    regardless of how large the dataset is. Same bundle shape every time,
    small dataset or large. Also bundles autods_cleaning_tools/ (see
    _write_cleaning_engine_bundle) whenever the notebook's cleaning cells
    will need to import from it."""
    import zipfile

    notebook_json, needs_cleaned_csv = nb_module.build_notebook(session, charts=charts)
    name = session.filename.rsplit(".", 1)[0]

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}_autods_notebook.ipynb", notebook_json)
        zf.writestr("original_data.csv", session.original_df.to_csv(index=False))
        if needs_cleaned_csv:
            zf.writestr("cleaned_data.csv", session.df.to_csv(index=False))
        if session.cleaning_log or session.chat_clean_log:
            _write_cleaning_engine_bundle(zf)
    zip_buf.seek(0)
    return zip_buf, f"{name}_autods_notebook_bundle.zip"


@app.get("/api/download_notebook/{session_id}")
def download_notebook(session_id: str):
    """Download a .ipynb notebook together with the CSV file(s) it reads
    from (original uncleaned data always, plus the cleaned result if any
    cleaning was done), zipped together so it runs immediately after
    unzipping with no setup step. Includes the full chart history if it was
    staged first via /api/stage_charts (falls back to just the most recent
    chart, session.last_visualization, otherwise)."""
    session = _get_session_or_404(session_id)
    buf, filename = _build_notebook_bundle_zip(session, charts=session.staged_charts)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/download_notebook/{session_id}")
def download_notebook_with_charts(session_id: str, req: ReportChartsRequest):
    """Same as the GET version, but takes the frontend's full chart history
    (customChartSpecs) so every visualization the user built gets its own
    cell in the notebook, not just the last one."""
    session = _get_session_or_404(session_id)
    if req.charts:
        session.staged_charts = req.charts
    buf, filename = _build_notebook_bundle_zip(session, charts=req.charts)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/download_notebook_form/{session_id}")
def download_notebook_form(session_id: str, charts_json: str = Form("[]")):
    """Form-submission counterpart to download_notebook_with_charts — see
    download_html_form's docstring for why this exists instead of the old
    stage-then-redirect-a-blank-tab approach."""
    session = _get_session_or_404(session_id)
    parsed_charts = _parse_form_charts(charts_json)
    if parsed_charts:
        session.staged_charts = parsed_charts
    buf, filename = _build_notebook_bundle_zip(session, charts=parsed_charts)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/download_notebook_form/{session_id}")
def download_notebook_form_get(session_id: str, charts_json: str = "[]"):
    """GET counterpart at the same URL, for the same reason as
    download_html_form_get — see its docstring."""
    session = _get_session_or_404(session_id)
    buf, filename = _build_notebook_bundle_zip(session, charts=_parse_form_charts(charts_json))
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
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
        headers={"Content-Disposition": f"attachment; filename={safe_name}_model.pkl", "Cache-Control": "no-store"},
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


@app.post("/api/session/purge")
def purge_sessions_endpoint():
    """Immediately deletes every orphaned .sessions/ file — i.e. any file not
    backing a session currently held in memory — regardless of age. The
    periodic sweep (see _periodic_session_cleanup) only removes files older
    than 7 days, which is fine for steady background hygiene but leaves
    same-day leftovers sitting around. This gives the frontend's "fresh
    reload" button a way to clear those out on demand, right when the user
    asks for it, instead of waiting out the week-long grace period."""
    removed = purge_stale_session_files(max_age_days=0)
    return {"ok": True, "removed": removed}


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
    _index_path = os.path.join(frontend_dir, "index.html")

    # Serve index.html explicitly (instead of leaving it to StaticFiles) so we
    # can force browsers to always revalidate it. This is a single-page app —
    # every fix ships by editing this one file in place, so a browser that
    # cached an old copy will keep running old (possibly buggy) JS
    # indefinitely with no visible sign anything is wrong, since the app
    # still "works", just with stale code. no-store forces a fresh fetch on
    # every load instead of relying on cache-validation round trips.
    @app.get("/", include_in_schema=False)
    def _serve_index():
        return FileResponse(_index_path, media_type="text/html", headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        })

    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")