"""
Simple in-memory session store with disk persistence.

Each uploaded dataset gets a session_id. We keep the raw dataframe,
any trained models, and metadata in memory for the life of the process.
DataFrames are also saved to disk (parquet) so sessions survive server
restarts — the frontend can restore a session even after a reload.
"""
import logging
import os
import uuid
import shutil
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger("autods.store")

# Directory where session dataframes are persisted between restarts
_SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".sessions")
_SESSION_DIR = os.path.abspath(_SESSION_DIR)
os.makedirs(_SESSION_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(_SESSION_DIR, f"{session_id}.csv")


def _original_path(session_id: str) -> str:
    return os.path.join(_SESSION_DIR, f"{session_id}_original.csv")


def _save_df_to_disk(session_id: str, df: pd.DataFrame, original_df: pd.DataFrame):
    """Save both current and original DataFrames to CSV."""
    try:
        df.to_csv(_session_path(session_id), index=False)
        original_df.to_csv(_original_path(session_id), index=False)
    except Exception:
        # Persistence is a safety net, not the source of truth (the in-memory
        # Session still has the data) — so we don't raise here. But a silent
        # failure meant a user could lose "unsaved" state on server restart
        # with zero indication why. Log it so it's at least visible.
        logger.exception("failed to persist session %s to disk", session_id)


def _load_df_from_disk(session_id: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load current and original DataFrames from disk. Returns None if not found."""
    try:
        p = _session_path(session_id)
        op = _original_path(session_id)
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        original_df = pd.read_csv(op) if os.path.exists(op) else df.copy()
        return df, original_df
    except Exception:
        logger.exception("failed to load session %s from disk", session_id)
        return None


def _delete_session_files(session_id: str):
    """Remove persisted files for a session."""
    for path in [_session_path(session_id), _original_path(session_id)]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.exception("failed to remove session file %s", path)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename duplicate column labels so every column name is unique.

    Duplicate column names are silently dangerous: pandas allows a DataFrame
    to have two (or more) columns sharing a name, but the moment that frame
    is serialized with .to_dict(orient="records") — which every preview,
    EDA profile, and filter response in this app does — the duplicate keys
    collapse into one, and only the LAST duplicate's value survives per row.
    If the first (real, non-null) column of that name happens to sit before
    a later duplicate that's empty, the preview shows that column as null
    for every row even though the actual first column is fully populated —
    with no error or warning surfaced to the user.

    Rather than chase every place a duplicate name could be introduced
    (raw CSVs with repeated headers, filtering, chat-driven column
    creation, joins, etc.), rename duplicates once here — the single
    place every dataframe passes through on its way into a session —
    so downstream code never has to special-case it.
    """
    cols = list(df.columns)
    if len(cols) == len(set(cols)):
        return df
    seen: dict[str, int] = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            candidate = f"{c}_{seen[c] + 1}"
            while candidate in seen:
                seen[c] += 1
                candidate = f"{c}_{seen[c] + 1}"
            seen[candidate] = 0
            new_cols.append(candidate)
    df = df.copy()
    df.columns = new_cols
    return df


class Session:
    UNDO_LIMIT = 20   # cap history depth so long cleaning sessions don't grow memory unbounded

    def __init__(self, df: pd.DataFrame, filename: str, session_id: str | None = None):
        self.id = session_id or str(uuid.uuid4())
        self.filename = filename
        df = _dedupe_columns(df)
        self.df = df
        self.original_df = df.copy()             # kept so chat cleaning commands can be reset
        self.target: Optional[str] = None
        self.problem_type: Optional[str] = None  # "classification" | "regression"
        self.models: Dict[str, Any] = {}        # name -> fitted pipeline
        self.leaderboard: list = []              # list of dicts (sorted)
        self.best_model_name: Optional[str] = None
        self.feature_columns: list = []
        self.label_encoder = None                # for classification target encoding
        self.cleaning_log: list = []
        self.chat_clean_log: list = []            # history of chat-based cleaning commands + results
        self.last_clean_options: dict = {}         # options used by the structured cleaning pipeline
        self.last_visualization: dict = {}         # latest chart payload produced by /api/visualize
        self._undo_stack: list = []               # snapshots of df taken before each cleaning change
        self.saved_runs: dict = {}                # run_id -> saved training snapshot (see main.py /api/train_runs)
        self.saved_predictions: dict = {}         # prediction_id -> prediction result snapshot
        self.unsupervised_results: dict = {}       # latest unsupervised analysis snapshots for reports
        self.progress_messages: list = []          # progress messages for long-running operations
        self.geojson: dict | None = None            # raw GeoJSON if a .geojson file was uploaded
        self.notes: str = ""                        # free-form user notes about this dataset, shown in downloads
        self.last_accessed: datetime = datetime.now()  # drives idle-eviction from memory; see evict_idle_sessions()
        # Persist to disk immediately on creation
        _save_df_to_disk(self.id, self.df, self.original_df)

    def touch(self):
        """Mark this session as recently used. Called on every get_session()
        lookup so evict_idle_sessions() only reaps sessions nobody's touched
        in a while, not ones actively being worked on."""
        self.last_accessed = datetime.now()

    def snapshot_before_change(self):
        """Call this right before mutating self.df, from any cleaning code path
        (structured 'Clean now', chat commands, type changes). Powers the Undo button."""
        self._undo_stack.append(self.df.copy())
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)

    def save_to_disk(self):
        """Persist the current df state to disk (called after any mutation)."""
        _save_df_to_disk(self.id, self.df, self.original_df)

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def undo(self) -> bool:
        """Restore the previous df state. Returns False if there was nothing to undo."""
        if not self._undo_stack:
            return False
        self.df = self._undo_stack.pop()
        self.save_to_disk()
        return True

    def _autosave_current_training(self):
        """If there's a trained model on the current data, snapshot it into
        saved_runs BEFORE it would otherwise be lost by clearing or replacing.

        This is the safety net that guarantees a model trained on one target
        (e.g. 'price') stays available for prediction even after the user moves
        on — training another target, cleaning the data, changing a column type,
        undoing a change, or running a chat action that modifies the dataframe.

        Idempotent: once models are cleared, repeated calls do nothing.
        """
        if not self.models:
            return
        if not self.target:
            return

        # Avoid piling up an identical auto-save if we're merely switching targets
        # and a previous identical run already exists (e.g. re-train same target).
        for run in self.saved_runs.values():
            if (
                run.get("auto_saved")
                and run.get("target") == self.target
                and run.get("best_model_name") == self.best_model_name
            ):
                return

        run_id = str(uuid.uuid4())
        self.saved_runs[run_id] = {
            "name": f"Auto-saved: {self.target}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "target": self.target,
            "problem_type": self.problem_type,
            "models": self.models,
            "leaderboard": self.leaderboard,
            "best_model_name": self.best_model_name,
            "label_encoder": self.label_encoder,
            "feature_columns": self.feature_columns,
            "auto_saved": True,   # marks this as an automatic snapshot (not a user-named run)
        }

    def clear_current_training(self, auto_preserve: bool = True):
        """Drop models tied to the previous current dataframe.

        Saved runs are intentionally kept: they are explicit model snapshots.
        When auto_preserve=True (the default), any currently trained model is
        first automatically snapshot into saved_runs so it is never lost —
        regardless of whether the clear is triggered by cleaning, a type change,
        undo, a chat action, or switching to a new target.

        Pass auto_preserve=False to skip the snapshot (e.g. when tearing down
        a session or intentionally discarding the model).
        """
        if auto_preserve:
            self._autosave_current_training()

        self.target = None
        self.problem_type = None
        self.models = {}
        self.leaderboard = []
        self.best_model_name = None
        self.feature_columns = []
        self.label_encoder = None
        self.progress_messages = []


SESSIONS: Dict[str, "Session"] = {}


def create_session(df: pd.DataFrame, filename: str) -> "Session":
    s = Session(df, filename)
    SESSIONS[s.id] = s
    return s


def get_session(session_id: str) -> "Session":
    # Fast path: already in memory
    if session_id in SESSIONS:
        s = SESSIONS[session_id]
        s.touch()
        return s

    # Slow path: try to restore from disk (survives server restarts)
    result = _load_df_from_disk(session_id)
    if result is None:
        raise KeyError("session_not_found")

    df, original_df = result
    # Reconstruct a minimal Session — models and undo history are lost,
    # but the data itself is intact so cleaning/training/EDA all work.
    s = Session.__new__(Session)
    s.id = session_id
    s.filename = "restored.parquet"
    s.df = df
    s.original_df = original_df
    s.target = None
    s.problem_type = None
    s.models = {}
    s.leaderboard = []
    s.best_model_name = None
    s.feature_columns = []
    s.label_encoder = None
    s.cleaning_log = []
    s.chat_clean_log = []
    s.last_clean_options = {}
    s.last_visualization = {}
    s._undo_stack = []
    s.saved_runs = {}
    s.saved_predictions = {}
    s.unsupervised_results = {}
    s.progress_messages = []
    s.geojson = None
    s.notes = ""
    s.last_accessed = datetime.now()

    SESSIONS[session_id] = s
    return s


def delete_session(session_id: str):
    """Remove a session from memory and disk."""
    SESSIONS.pop(session_id, None)
    _delete_session_files(session_id)


def evict_idle_sessions(max_idle_hours: int = 24) -> int:
    """Drop sessions from the in-memory SESSIONS dict that haven't been
    accessed (via get_session) in over max_idle_hours.

    This only frees memory — it does NOT touch the CSV files on disk, so an
    evicted session is still transparently restored (via the slow path in
    get_session) the next time the frontend asks for it; it just won't hold
    its full in-memory state (trained models, undo history) if it was never
    otherwise saved. Disk files themselves are cleaned up separately, much
    later, by purge_stale_session_files. Returns the number of sessions
    evicted from memory.
    """
    cutoff = datetime.now().timestamp() - (max_idle_hours * 3600)
    stale_ids = [
        sid for sid, s in SESSIONS.items()
        if getattr(s, "last_accessed", None) is not None
        and s.last_accessed.timestamp() < cutoff
    ]
    for sid in stale_ids:
        SESSIONS.pop(sid, None)
    return len(stale_ids)


def purge_stale_session_files(max_age_days: int = 7) -> int:
    """Delete .sessions/ CSV files that are (a) older than max_age_days AND
    (b) not backing any session currently held in memory. This cleans up
    files left behind by dataset tabs that were closed before the delete
    endpoint existed, or sessions orphaned by a server crash/restart.
    Returns the number of files removed."""
    removed = 0
    cutoff = datetime.now().timestamp() - (max_age_days * 86400)
    try:
        active_ids = set(SESSIONS.keys())
        for fname in os.listdir(_SESSION_DIR):
            if not fname.endswith(".csv"):
                continue
            session_id = fname.replace("_original.csv", "").replace(".csv", "")
            if session_id in active_ids:
                continue
            fpath = os.path.join(_SESSION_DIR, fname)
            try:
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    removed += 1
            except OSError:
                pass
    except Exception:
        pass
    return removed