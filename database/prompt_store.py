"""
database/prompt_store.py
Member 4: Backend & Database Engineer
-------------------------------------
Thread-safe SQLite repository providing CRUD operations, immutable version
history, version diff/restore, collections, user feedback tracking, and
real-time dashboard analytics for Intelligent Prompt Optimization Assistant 2.0.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from database.models import (
    CollectionRecord,
    FeedbackRecord,
    PromptRecord,
    PromptVersionRecord,
    run_safe_migrations,
    safe_json_loads,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "prompt_vault.db"


class PromptStore:
    """Persistent SQLite repository for prompts, versions, collections, and feedback."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        env_path = os.getenv("DATABASE_PATH")
        self.db_path = Path(db_path or env_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a SQLite connection with Row factory and foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        """Run non-destructive schema setup and migrations."""
        with self._connect() as conn:
            run_safe_migrations(conn)
        logger.info("PromptStore initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Activity Telemetry (Drives Real Analytics Dashboard)
    # ------------------------------------------------------------------

    def log_activity(
        self,
        event_type: str,
        category: str = "General",
        original_score: float = 0.0,
        optimized_score: float = 0.0,
        security_risk: str = "Low",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an analysis, optimization, security scan, or playground run."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO activity_logs
                (event_type, category, original_score, optimized_score, security_risk, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    category or "General",
                    float(original_score),
                    float(optimized_score),
                    security_risk or "Low",
                    json.dumps(metadata or {}),
                    utc_now_iso(),
                ),
            )

    # ------------------------------------------------------------------
    # Collections Management
    # ------------------------------------------------------------------

    def create_collection(
        self, name: str, description: str = "", color: str = "#2788F5"
    ) -> CollectionRecord:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Collection name cannot be empty.")

        now = utc_now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO collections (name, description, color, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_name, description.strip(), color, now),
            )
            return CollectionRecord(
                id=cursor.lastrowid,
                name=clean_name,
                description=description.strip(),
                color=color,
                created_at=now,
            )

    def list_collections(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, COUNT(p.id) AS prompt_count
                FROM collections c
                LEFT JOIN prompts p ON p.collection_id = c.id
                GROUP BY c.id
                ORDER BY c.id ASC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Prompt CRUD & Version Control
    # ------------------------------------------------------------------

    def _row_to_prompt(self, row: sqlite3.Row) -> PromptRecord:
        keys = row.keys()
        return PromptRecord(
            id=row["id"],
            title=row["title"] if "title" in keys and row["title"] else "Saved Prompt",
            original_prompt=row["original_prompt"],
            optimized_prompt=row["optimized_prompt"],
            category=row["category"] or "General",
            target_model=row["target_model"] if "target_model" in keys else "gemini-2.5-flash",
            collection_id=row["collection_id"] if "collection_id" in keys else 1,
            original_score=float(row["original_score"] or 0.0),
            optimized_score=float(row["optimized_score"] or 0.0),
            cnn_prediction=row["cnn_prediction"] if "cnn_prediction" in keys else "Unknown",
            bilstm_prediction=row["bilstm_prediction"] if "bilstm_prediction" in keys else "Unknown",
            dimension_scores=safe_json_loads(
                row["dimension_scores_json"] if "dimension_scores_json" in keys else "{}", {}
            ),
            weaknesses=safe_json_loads(
                row["weaknesses_json"] if "weaknesses_json" in keys else "[]", []
            ),
            strategy_applied=row["strategy_applied"] if "strategy_applied" in keys else "",
            security_risk=row["security_risk"] if "security_risk" in keys else "Low",
            current_version=int(row["current_version"] or 1) if "current_version" in keys else 1,
            optimization_settings=safe_json_loads(
                row["optimization_settings_json"] if "optimization_settings_json" in keys else "{}",
                {},
            ),
            tags=safe_json_loads(row["tags_json"] if "tags_json" in keys else "[]", []),
            created_at=row["created_at"],
            updated_at=row["updated_at"] if "updated_at" in keys and row["updated_at"] else row["created_at"],
        )

    def save_prompt(self, record: PromptRecord, change_summary: str = "Initial optimization") -> PromptRecord:
        """Save a new prompt into the library and create Version 1 in version history."""
        now = utc_now_iso()
        record.created_at = now
        record.updated_at = now
        record.current_version = 1

        if not record.title or record.title == "Untitled Prompt":
            snippet = record.original_prompt.strip().splitlines()[0][:48] if record.original_prompt.strip() else "Prompt"
            record.title = f"{snippet}..." if len(snippet) >= 48 else snippet

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prompts (
                    title, original_prompt, optimized_prompt, category, target_model,
                    collection_id, original_score, optimized_score, cnn_prediction,
                    bilstm_prediction, dimension_scores_json, weaknesses_json,
                    strategy_applied, security_risk, current_version,
                    optimization_settings_json, tags_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.title,
                    record.original_prompt,
                    record.optimized_prompt,
                    record.category,
                    record.target_model,
                    record.collection_id or 1,
                    float(record.original_score),
                    float(record.optimized_score),
                    record.cnn_prediction,
                    record.bilstm_prediction,
                    json.dumps(record.dimension_scores),
                    json.dumps(record.weaknesses),
                    record.strategy_applied,
                    record.security_risk,
                    1,
                    json.dumps(record.optimization_settings),
                    json.dumps(record.tags),
                    now,
                    now,
                ),
            )
            record.id = cursor.lastrowid

            # Record Version 1 snapshot
            conn.execute(
                """
                INSERT INTO prompt_versions (
                    prompt_id, version_number, prompt_text, system_notes,
                    quality_score, cnn_label, bilstm_label, strategy_used,
                    change_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    1,
                    record.optimized_prompt,
                    f"Original: {record.original_prompt[:180]}",
                    float(record.optimized_score),
                    record.cnn_prediction,
                    record.bilstm_prediction,
                    record.strategy_applied,
                    change_summary,
                    now,
                ),
            )

        return record

    def create_new_version(
        self,
        prompt_id: int,
        new_prompt_text: str,
        quality_score: float = 0.0,
        cnn_label: str = "Unknown",
        bilstm_label: str = "Unknown",
        strategy_used: str = "",
        change_summary: str = "Updated revision",
    ) -> PromptVersionRecord:
        """Append a new version to an existing prompt and advance current_version."""
        now = utc_now_iso()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT current_version FROM prompts WHERE id = ?", (prompt_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Prompt ID {prompt_id} not found.")

            next_ver = int(row["current_version"] or 1) + 1
            cursor = conn.execute(
                """
                INSERT INTO prompt_versions (
                    prompt_id, version_number, prompt_text, system_notes,
                    quality_score, cnn_label, bilstm_label, strategy_used,
                    change_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt_id,
                    next_ver,
                    new_prompt_text,
                    "",
                    float(quality_score),
                    cnn_label,
                    bilstm_label,
                    strategy_used,
                    change_summary,
                    now,
                ),
            )

            conn.execute(
                """
                UPDATE prompts
                SET optimized_prompt = ?,
                    optimized_score = ?,
                    strategy_applied = CASE WHEN ? != '' THEN ? ELSE strategy_applied END,
                    current_version = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    new_prompt_text,
                    float(quality_score),
                    strategy_used,
                    strategy_used,
                    next_ver,
                    now,
                    prompt_id,
                ),
            )

            return PromptVersionRecord(
                id=cursor.lastrowid,
                prompt_id=prompt_id,
                version_number=next_ver,
                prompt_text=new_prompt_text,
                quality_score=float(quality_score),
                cnn_label=cnn_label,
                bilstm_label=bilstm_label,
                strategy_used=strategy_used,
                change_summary=change_summary,
                created_at=now,
            )

    def get_versions(self, prompt_id: int) -> List[PromptVersionRecord]:
        """Return all saved versions for a prompt in descending version order."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM prompt_versions
                WHERE prompt_id = ?
                ORDER BY version_number DESC
                """,
                (prompt_id,),
            ).fetchall()

            # Backfill Version 1 for legacy v1.0 prompts that have no version rows yet
            if not rows:
                prompt_row = conn.execute(
                    "SELECT * FROM prompts WHERE id = ?", (prompt_id,)
                ).fetchone()
                if prompt_row:
                    ver1 = PromptVersionRecord(
                        prompt_id=prompt_id,
                        version_number=1,
                        prompt_text=prompt_row["optimized_prompt"],
                        quality_score=float(prompt_row["optimized_score"] or 0.0),
                        strategy_used=prompt_row["strategy_applied"] or "",
                        change_summary="Legacy v1 baseline",
                        created_at=prompt_row["created_at"],
                    )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO prompt_versions (
                            prompt_id, version_number, prompt_text, system_notes,
                            quality_score, cnn_label, bilstm_label, strategy_used,
                            change_summary, created_at
                        ) VALUES (?, 1, ?, '', ?, 'Unknown', 'Unknown', ?, 'Legacy v1 baseline', ?)
                        """,
                        (
                            prompt_id,
                            ver1.prompt_text,
                            ver1.quality_score,
                            ver1.strategy_used,
                            ver1.created_at,
                        ),
                    )
                    return [ver1]

            return [
                PromptVersionRecord(
                    id=r["id"],
                    prompt_id=r["prompt_id"],
                    version_number=r["version_number"],
                    prompt_text=r["prompt_text"],
                    system_notes=r["system_notes"] or "",
                    quality_score=float(r["quality_score"] or 0.0),
                    cnn_label=r["cnn_label"] or "Unknown",
                    bilstm_label=r["bilstm_label"] or "Unknown",
                    strategy_used=r["strategy_used"] or "",
                    change_summary=r["change_summary"] or "",
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def restore_version(self, prompt_id: int, version_number: int) -> PromptVersionRecord:
        """Restore a historical version as a new head revision so no history is lost."""
        versions = self.get_versions(prompt_id)
        target = next((v for v in versions if v.version_number == version_number), None)
        if not target:
            raise ValueError(f"Version v{version_number} not found for prompt {prompt_id}.")

        return self.create_new_version(
            prompt_id=prompt_id,
            new_prompt_text=target.prompt_text,
            quality_score=target.quality_score,
            cnn_label=target.cnn_label,
            bilstm_label=target.bilstm_label,
            strategy_used=target.strategy_used,
            change_summary=f"Restored from v{version_number}",
        )

    def get_prompt(self, prompt_id: int) -> Optional[PromptRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
            return self._row_to_prompt(row) if row else None

    def list_prompts(
        self,
        search_query: str = "",
        category: Optional[str] = None,
        collection_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[PromptRecord]:
        clauses: List[str] = []
        params: List[Any] = []

        if search_query.strip():
            clauses.append(
                "(title LIKE ? OR original_prompt LIKE ? OR optimized_prompt LIKE ? OR tags_json LIKE ?)"
            )
            term = f"%{search_query.strip()}%"
            params.extend([term, term, term, term])

        if category and category != "All":
            clauses.append("category = ?")
            params.append(category)

        if collection_id is not None and collection_id > 0:
            clauses.append("collection_id = ?")
            params.append(collection_id)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM prompts {where_sql} ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_prompt(r) for r in rows]

    def delete_prompt(self, prompt_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Feedback Persistence & Training Export
    # ------------------------------------------------------------------

    def save_feedback(self, feedback: FeedbackRecord) -> FeedbackRecord:
        now = utc_now_iso()
        feedback.created_at = now
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prompt_feedback (
                    prompt_id, original_prompt, optimized_prompt, thumbs_up,
                    rating, comment, detected_weaknesses_json, strategy_used, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.prompt_id,
                    feedback.original_prompt,
                    feedback.optimized_prompt,
                    1 if feedback.thumbs_up else 0,
                    max(1, min(5, int(feedback.rating))),
                    feedback.comment.strip(),
                    json.dumps(feedback.detected_weaknesses),
                    feedback.strategy_used,
                    now,
                ),
            )
            feedback.id = cursor.lastrowid
        return feedback

    def export_prompts(self, fmt: str = "json") -> str:
        """Export saved prompts in JSON, Markdown, or CSV format."""
        prompts = self.list_prompts(limit=1000)
        if fmt.lower() == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "id",
                    "title",
                    "category",
                    "original_score",
                    "optimized_score",
                    "cnn_prediction",
                    "bilstm_prediction",
                    "strategy_applied",
                    "security_risk",
                    "original_prompt",
                    "optimized_prompt",
                    "created_at",
                ]
            )
            for p in prompts:
                writer.writerow(
                    [
                        p.id,
                        p.title,
                        p.category,
                        p.original_score,
                        p.optimized_score,
                        p.cnn_prediction,
                        p.bilstm_prediction,
                        p.strategy_applied,
                        p.security_risk,
                        p.original_prompt,
                        p.optimized_prompt,
                        p.created_at,
                    ]
                )
            return buf.getvalue()

        if fmt.lower() == "markdown":
            lines = ["# Intelligent Prompt Optimization Assistant — Exported Library\n"]
            for p in prompts:
                lines.append(f"## {p.title} (v{p.current_version} | {p.category})")
                lines.append(
                    f"- **Quality Score:** {p.original_score:.1f} → **{p.optimized_score:.1f}** "
                    f"| **CNN:** {p.cnn_prediction} | **Bi-LSTM:** {p.bilstm_prediction} | **Security:** {p.security_risk}"
                )
                lines.append(f"\n### Original Prompt\n```text\n{p.original_prompt}\n```")
                lines.append(f"\n### Optimized Prompt\n```text\n{p.optimized_prompt}\n```\n---\n")
            return "\n".join(lines)

        return json.dumps([p.to_dict() for p in prompts], indent=2)

    def export_feedback_jsonl(self) -> str:
        """Export user feedback records as JSONL suitable for future offline model training."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM prompt_feedback ORDER BY created_at DESC"
            ).fetchall()
            lines = []
            for r in rows:
                record = {
                    "original_prompt": r["original_prompt"],
                    "optimized_prompt": r["optimized_prompt"],
                    "thumbs_up": bool(r["thumbs_up"]),
                    "rating": int(r["rating"]),
                    "comment": r["comment"],
                    "detected_weaknesses": safe_json_loads(r["detected_weaknesses_json"], []),
                    "strategy_used": r["strategy_used"],
                    "timestamp": r["created_at"],
                }
                lines.append(json.dumps(record))
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Live Dashboard Analytics (Zero Fabricated Numbers)
    # ------------------------------------------------------------------

    def get_dashboard_analytics(self) -> Dict[str, Any]:
        """Compute genuine platform metrics from stored prompts, logs, and feedback."""
        with self._connect() as conn:
            prompts = [self._row_to_prompt(r) for r in conn.execute("SELECT * FROM prompts").fetchall()]
            logs = conn.execute("SELECT * FROM activity_logs ORDER BY created_at DESC").fetchall()
            feedback_rows = conn.execute("SELECT * FROM prompt_feedback ORDER BY created_at ASC").fetchall()

        total_saved = len(prompts)
        total_analyzed = max(total_saved, len(logs))
        total_optimized = max(
            total_saved,
            sum(1 for l in logs if l["event_type"] == "optimize"),
        )

        # Quality averages
        if prompts:
            avg_quality = round(sum(p.optimized_score for p in prompts) / total_saved, 2)
            deltas = [p.optimized_score - p.original_score for p in prompts]
            avg_improvement = round(sum(deltas) / len(deltas), 2)
            improved_count = sum(1 for d in deltas if d > 0)
            success_rate = round((improved_count / total_saved) * 100.0, 1)
        elif logs:
            opt_logs = [l for l in logs if l["optimized_score"] > 0]
            avg_quality = round(
                sum(l["optimized_score"] for l in opt_logs) / len(opt_logs), 2
            ) if opt_logs else 0.0
            deltas = [l["optimized_score"] - l["original_score"] for l in opt_logs]
            avg_improvement = round(sum(deltas) / len(deltas), 2) if deltas else 0.0
            success_rate = round(
                (sum(1 for d in deltas if d > 0) / len(opt_logs)) * 100.0, 1
            ) if opt_logs else 0.0
        else:
            avg_quality = 0.0
            avg_improvement = 0.0
            success_rate = 0.0

        # Security warnings count (Medium or High risk)
        prompt_warnings = sum(1 for p in prompts if p.security_risk in ("Medium", "High"))
        log_warnings = sum(
            1 for l in logs if l["event_type"] == "security_scan" and l["security_risk"] in ("Medium", "High")
        )
        security_warnings = prompt_warnings + log_warnings

        # Category distribution
        category_counts = Counter(p.category for p in prompts if p.category)
        if not category_counts and logs:
            category_counts = Counter(l["category"] for l in logs if l["category"])

        # Recurring weaknesses across saved prompts and feedback
        weakness_counter: Counter[str] = Counter()
        for p in prompts:
            weakness_counter.update(p.weaknesses)
        for fb in feedback_rows:
            weakness_counter.update(safe_json_loads(fb["detected_weaknesses_json"], []))

        # Feedback trends
        total_feedback = len(feedback_rows)
        positive_feedback = sum(1 for fb in feedback_rows if fb["thumbs_up"] == 1)
        avg_rating = (
            round(sum(int(fb["rating"]) for fb in feedback_rows) / total_feedback, 2)
            if total_feedback
            else 0.0
        )
        feedback_history = [
            {
                "index": idx + 1,
                "rating": int(fb["rating"]),
                "thumbs_up": bool(fb["thumbs_up"]),
                "created_at": fb["created_at"][:10],
            }
            for idx, fb in enumerate(feedback_rows[-30:])
        ]

        # Recent activity feed
        recent_activity = [
            {
                "id": p.id,
                "title": p.title,
                "category": p.category,
                "original_score": round(p.original_score, 1),
                "optimized_score": round(p.optimized_score, 1),
                "delta": round(p.optimized_score - p.original_score, 1),
                "security_risk": p.security_risk,
                "updated_at": p.updated_at,
            }
            for p in sorted(prompts, key=lambda x: x.updated_at, reverse=True)[:8]
        ]

        return {
            "total_analyzed": total_analyzed,
            "total_optimized": total_optimized,
            "total_saved": total_saved,
            "avg_quality": avg_quality,
            "avg_improvement": avg_improvement,
            "security_warnings": security_warnings,
            "success_rate": success_rate,
            "category_distribution": dict(category_counts),
            "top_weaknesses": dict(weakness_counter.most_common(6)),
            "feedback_summary": {
                "total": total_feedback,
                "positive": positive_feedback,
                "negative": total_feedback - positive_feedback,
                "avg_rating": avg_rating,
                "history": feedback_history,
            },
            "recent_activity": recent_activity,
        }