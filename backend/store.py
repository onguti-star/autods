"""
Simple in-memory session store.

Each uploaded dataset gets a session_id. We keep the raw dataframe,
any trained models, and metadata in memory for the life of the process.
This is fine for a prototype / single-user local run. For production,
swap this for Redis + object storage (S3) keyed by session_id.
"""
import uuid
from typing import Any, Dict, Optional

import pandas as pd


class Session:
    UNDO_LIMIT = 20   # cap history depth so long cleaning sessions don't grow memory unbounded

    def __init__(self, df: pd.DataFrame, filename: str):
        self.id = str(uuid.uuid4())
        self.filename = filename
        self.df = df
        self.original_df = df.copy()             # kept so chat cleaning commands can be reset
        self.target: Optional[str] = None
        self.problem_type: Optional[str] = None  # "classification" | "regression"
        self.models: Dict[str, Any] = {}        # name -> fitted pipeline
        self.leaderboard: list = []              # list of dicts (sorted)
        self.best_model_name: Optional[str] = None
        self.feature_importance: list = []        # latest best-model feature importance rows
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
        self.pca_result: dict = {}                 # latest PCA analysis snapshot for reports
        self.progress_messages: list = []          # progress messages for long-running operations

    def snapshot_before_change(self):
        """Call this right before mutating self.df, from any cleaning code path
        (structured 'Clean now', chat commands, type changes). Powers the Undo button."""
        self._undo_stack.append(self.df.copy())
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def undo(self) -> bool:
        """Restore the previous df state. Returns False if there was nothing to undo."""
        if not self._undo_stack:
            return False
        self.df = self._undo_stack.pop()
        return True

    def clear_current_training(self):
        """Drop models tied to the previous current dataframe.

        Saved runs are intentionally kept: they are explicit model snapshots.
        """
        self.target = None
        self.problem_type = None
        self.models = {}
        self.leaderboard = []
        self.best_model_name = None
        self.feature_importance = []
        self.feature_columns = []
        self.label_encoder = None
        self.pca_result = {}
        self.progress_messages = []


SESSIONS: Dict[str, Session] = {}


def create_session(df: pd.DataFrame, filename: str) -> Session:
    s = Session(df, filename)
    SESSIONS[s.id] = s
    return s


def get_session(session_id: str) -> Session:
    if session_id not in SESSIONS:
        raise KeyError("session_not_found")
    return SESSIONS[session_id]
