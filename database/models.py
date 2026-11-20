"""
database/models.py
Member 4: Backend & Database Engineer
-------------------------------------
SQLite schema definitions, data transfer dataclasses, and non-destructive
schema migration helpers for Intelligent Prompt Optimization Assistant 2.0.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CollectionRecord:
    """Represents a user-created folder/collection for organizing prompts."""
    id: Optional[int] = None
    name: str = "General"
    description: str = ""
    color: str = "#2788F5"
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptVersionRecord:
    """Immutable snapshot of a prompt at a specific revision number."""
    id: Optional[int] = None
    prompt_id: int = 0
    version_number: int = 1
    prompt_text: str = ""
    system_notes: str = ""
    quality_score: float = 0.0
    cnn_label: str = "Unknown"
    bilstm_label: str = "Unknown"
    strategy_used: str = ""
    change_summary: str = "Initial version"
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptRecord:
    """Primary saved prompt entity with version tracking and DL metadata."""
    id: Optional[int] = None
    title: str = "Untitled Prompt"
    original_prompt: str = ""
    optimized_prompt: str = ""
    category: str = "General"
    target_model: str = "gemini-2.5-flash"
    collection_id: Optional[int] = None
    original_score: float = 0.0
    optimized_score: float = 0.0
    cnn_prediction: str = "Unknown"
    bilstm_prediction: str = "Unknown"
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    weaknesses: List[str] = field(default_factory=list)
    strategy_applied: str = ""
    security_risk: str = "Low"
    current_version: int = 1
    optimization_settings: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackRecord:
    """User evaluation of an optimization run for quality tracking & future training."""
    id: Optional[int] = None
    prompt_id: Optional[int] = None
    original_prompt: str = ""
    optimized_prompt: str = ""
    thumbs_up: bool = True
    rating: int = 5
    comment: str = ""
    detected_weaknesses: List[str] = field(default_factory=list)
    strategy_used: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


TABLES_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    color TEXT DEFAULT '#2788F5',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    original_prompt TEXT NOT NULL,
    optimized_prompt TEXT NOT NULL,
    category TEXT DEFAULT 'General',
    target_model TEXT DEFAULT 'gemini-2.5-flash',
    collection_id INTEGER REFERENCES collections(id) ON DELETE SET NULL,
    original_score REAL DEFAULT 0.0,
    optimized_score REAL DEFAULT 0.0,
    cnn_prediction TEXT DEFAULT 'Unknown',
    bilstm_prediction TEXT DEFAULT 'Unknown',
    dimension_scores_json TEXT DEFAULT '{}',
    weaknesses_json TEXT DEFAULT '[]',
    strategy_applied TEXT DEFAULT '',
    security_risk TEXT DEFAULT 'Low',
    current_version INTEGER DEFAULT 1,
    optimization_settings_json TEXT DEFAULT '{}',
    tags_json TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id INTEGER NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    prompt_text TEXT NOT NULL DEFAULT '',
    system_notes TEXT DEFAULT '',
    quality_score REAL DEFAULT 0.0,
    cnn_label TEXT DEFAULT 'Unknown',
    bilstm_label TEXT DEFAULT 'Unknown',
    strategy_used TEXT DEFAULT '',
    change_summary TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(prompt_id, version_number)
);

CREATE TABLE IF NOT EXISTS prompt_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id INTEGER REFERENCES prompts(id) ON DELETE SET NULL,
    original_prompt TEXT DEFAULT '',
    optimized_prompt TEXT DEFAULT '',
    thumbs_up INTEGER NOT NULL DEFAULT 1,
    rating INTEGER NOT NULL DEFAULT 5,
    comment TEXT DEFAULT '',
    detected_weaknesses_json TEXT DEFAULT '[]',
    strategy_used TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    category TEXT DEFAULT 'General',
    original_score REAL DEFAULT 0.0,
    optimized_score REAL DEFAULT 0.0,
    security_risk TEXT DEFAULT 'Low',
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_prompts_category ON prompts(category);
CREATE INDEX IF NOT EXISTS idx_prompts_collection ON prompts(collection_id);
CREATE INDEX IF NOT EXISTS idx_versions_prompt ON prompt_versions(prompt_id);
CREATE INDEX IF NOT EXISTS idx_feedback_prompt ON prompt_feedback(prompt_id);
CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_logs(created_at);
"""


def _ensure_columns(conn: sqlite3.Connection, table_name: str, columns: Dict[str, str]) -> set[str]:
    """Inspect a SQLite table and safely add any missing columns."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}
    for col_name, col_def in columns.items():
        if col_name not in existing:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}")
            existing.add(col_name)
    return existing


def run_safe_migrations(conn: sqlite3.Connection) -> None:
    """Create missing v2.0 tables and append missing columns to existing v1.0 tables."""
    conn.executescript(TABLES_SQL)

    prompt_cols = _ensure_columns(
        conn,
        "prompts",
        {
            "title": "TEXT DEFAULT 'Saved Prompt'",
            "category": "TEXT DEFAULT 'General'",
            "target_model": "TEXT DEFAULT 'gemini-2.5-flash'",
            "collection_id": "INTEGER",
            "original_score": "REAL DEFAULT 0.0",
            "optimized_score": "REAL DEFAULT 0.0",
            "cnn_prediction": "TEXT DEFAULT 'Unknown'",
            "bilstm_prediction": "TEXT DEFAULT 'Unknown'",
            "dimension_scores_json": "TEXT DEFAULT '{}'",
            "weaknesses_json": "TEXT DEFAULT '[]'",
            "strategy_applied": "TEXT DEFAULT ''",
            "security_risk": "TEXT DEFAULT 'Low'",
            "current_version": "INTEGER DEFAULT 1",
            "optimization_settings_json": "TEXT DEFAULT '{}'",
            "tags_json": "TEXT DEFAULT '[]'",
            "updated_at": f"TEXT DEFAULT '{utc_now_iso()}'",
        },
    )

    if "initial_score" in prompt_cols:
        conn.execute("UPDATE prompts SET original_score = COALESCE(initial_score, 0.0) WHERE original_score = 0.0")
    elif "before_score" in prompt_cols:
        conn.execute("UPDATE prompts SET original_score = COALESCE(before_score, 0.0) WHERE original_score = 0.0")

    if "final_score" in prompt_cols:
        conn.execute("UPDATE prompts SET optimized_score = COALESCE(final_score, 0.0) WHERE optimized_score = 0.0")
    elif "after_score" in prompt_cols:
        conn.execute("UPDATE prompts SET optimized_score = COALESCE(after_score, 0.0) WHERE optimized_score = 0.0")

    version_cols = _ensure_columns(
        conn,
        "prompt_versions",
        {
            "prompt_id": "INTEGER",
            "version_number": "INTEGER DEFAULT 1",
            "prompt_text": "TEXT DEFAULT ''",
            "system_notes": "TEXT DEFAULT ''",
            "quality_score": "REAL DEFAULT 0.0",
            "cnn_label": "TEXT DEFAULT 'Unknown'",
            "bilstm_label": "TEXT DEFAULT 'Unknown'",
            "strategy_used": "TEXT DEFAULT ''",
            "change_summary": "TEXT DEFAULT ''",
            "created_at": f"TEXT DEFAULT '{utc_now_iso()}'",
        },
    )

    for legacy_col in ("optimized_prompt", "content", "text", "prompt"):
        if legacy_col in version_cols:
            conn.execute(
                f"UPDATE prompt_versions SET prompt_text = COALESCE({legacy_col}, '') WHERE prompt_text = ''"
            )
            break

    _ensure_columns(
        conn,
        "prompt_feedback",
        {
            "prompt_id": "INTEGER",
            "original_prompt": "TEXT DEFAULT ''",
            "optimized_prompt": "TEXT DEFAULT ''",
            "thumbs_up": "INTEGER NOT NULL DEFAULT 1",
            "rating": "INTEGER NOT NULL DEFAULT 5",
            "comment": "TEXT DEFAULT ''",
            "detected_weaknesses_json": "TEXT DEFAULT '[]'",
            "strategy_used": "TEXT DEFAULT ''",
            "created_at": f"TEXT DEFAULT '{utc_now_iso()}'",
        },
    )

    _ensure_columns(
        conn,
        "activity_logs",
        {
            "event_type": "TEXT DEFAULT 'optimize'",
            "category": "TEXT DEFAULT 'General'",
            "original_score": "REAL DEFAULT 0.0",
            "optimized_score": "REAL DEFAULT 0.0",
            "security_risk": "TEXT DEFAULT 'Low'",
            "metadata_json": "TEXT DEFAULT '{}'",
            "created_at": f"TEXT DEFAULT '{utc_now_iso()}'",
        },
    )

    conn.executescript(INDEXES_SQL)

    conn.execute(
        """
        INSERT OR IGNORE INTO collections (id, name, description, color, created_at)
        VALUES (1, 'General Workspace', 'Default prompt repository', '#2788F5', ?)
        """,
        (utc_now_iso(),),
    )
    conn.commit()


def safe_json_loads(raw: Optional[str], default: Any) -> Any:
    """Parse JSON column safely, returning default on malformed input."""
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default