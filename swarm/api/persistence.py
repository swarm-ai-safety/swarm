"""SQLite-backed persistence for Agent API runs and posts.

Replaces the in-memory dicts with a durable store that survives restarts.
Uses a write-through cache for active (non-terminal) runs so hot-path
reads don't hit disk.

Schema is created lazily on first access (no migration tooling needed).
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from swarm.api.models.post import PostResponse
from swarm.api.models.run import (
    RunResponse,
    RunStatus,
    RunSummaryMetrics,
    RunVisibility,
)

logger = logging.getLogger(__name__)

# Default DB path — sits next to the runs/ directory at repo root.
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "runs" / "agent_api.db"

_SCHEMA_VERSION = 1

_CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    scenario_id   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued',
    visibility    TEXT NOT NULL DEFAULT 'private',
    agent_id      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    completed_at  TEXT,
    progress      REAL DEFAULT 0.0,
    summary_json  TEXT,
    status_url    TEXT NOT NULL,
    public_url    TEXT,
    error         TEXT
);
"""

_CREATE_POSTS_TABLE = """
CREATE TABLE IF NOT EXISTS posts (
    post_id       TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    agent_id      TEXT NOT NULL,
    title         TEXT NOT NULL,
    blurb         TEXT NOT NULL,
    key_metrics   TEXT NOT NULL DEFAULT '{}',
    tags          TEXT NOT NULL DEFAULT '[]',
    published_at  TEXT NOT NULL,
    run_url       TEXT,
    upvotes       INTEGER NOT NULL DEFAULT 0,
    downvotes     INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_VOTES_TABLE = """
CREATE TABLE IF NOT EXISTS votes (
    vote_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id   TEXT NOT NULL REFERENCES posts(post_id),
    agent_id  TEXT NOT NULL,
    direction INTEGER NOT NULL,  -- +1 upvote, -1 downvote
    voted_at  TEXT NOT NULL,
    UNIQUE(post_id, agent_id)
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);",
    "CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_posts_published ON posts(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_posts_agent ON posts(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_votes_post ON votes(post_id);",
]


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class RunStore:
    """SQLite-backed run storage with in-memory cache for active runs."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Write-through cache: only active (non-terminal) runs stay here.
        self._cache: dict[str, RunResponse] = {}

        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_RUNS_TABLE)
            for idx in _CREATE_INDEXES:
                if "runs" in idx:
                    conn.execute(idx)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, run: RunResponse) -> None:
        """Upsert a run into both cache and DB.

        Lock covers both cache and DB write to prevent inconsistency
        (security fix 1.7).
        """
        summary_json = (
            run.summary_metrics.model_dump_json() if run.summary_metrics else None
        )

        with self._lock:
            # Update cache for active runs
            terminal = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
            if run.status not in terminal:
                self._cache[run.run_id] = run
            else:
                self._cache.pop(run.run_id, None)

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, scenario_id, status, visibility, agent_id,
                         created_at, started_at, completed_at, progress,
                         summary_json, status_url, public_url, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status=excluded.status,
                        started_at=excluded.started_at,
                        completed_at=excluded.completed_at,
                        progress=excluded.progress,
                        summary_json=excluded.summary_json,
                        error=excluded.error
                    """,
                    (
                        run.run_id,
                        run.scenario_id,
                        run.status.value,
                        run.visibility.value,
                        run.agent_id,
                        _iso(run.created_at),
                        _iso(run.started_at),
                        _iso(run.completed_at),
                        run.progress,
                        summary_json,
                        run.status_url,
                        run.public_url,
                        run.error,
                    ),
                )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> Optional[RunResponse]:
        """Fetch a run by ID. Checks cache first, then DB."""
        with self._lock:
            cached = self._cache.get(run_id)
            if cached is not None:
                return cached

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def list_by_agent(self, agent_id: str) -> list[RunResponse]:
        """List all runs for an agent (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE agent_id = ? ORDER BY created_at DESC",
                (agent_id,),
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def count_active(self, agent_id: str) -> int:
        """Count queued + running runs for an agent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE agent_id = ? AND status IN ('queued', 'running')",
                (agent_id,),
            ).fetchone()
        return row[0] if row else 0

    def total_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return row[0] if row else 0

    def get_multiple(self, run_ids: list[str]) -> list[RunResponse]:
        """Fetch multiple runs by IDs."""
        if not run_ids:
            return []
        placeholders = ",".join("?" * len(run_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs WHERE run_id IN ({placeholders})",
                run_ids,
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_run(self, row: sqlite3.Row) -> RunResponse:
        summary = None
        if row["summary_json"]:
            try:
                summary = RunSummaryMetrics.model_validate_json(row["summary_json"])
            except Exception:
                logger.warning("Corrupt summary_json for run %s", row["run_id"])
        return RunResponse(
            run_id=row["run_id"],
            scenario_id=row["scenario_id"],
            status=RunStatus(row["status"]),
            visibility=RunVisibility(row["visibility"]),
            agent_id=row["agent_id"],
            created_at=_parse_dt(row["created_at"]),  # type: ignore[arg-type]
            started_at=_parse_dt(row["started_at"]),
            completed_at=_parse_dt(row["completed_at"]),
            progress=row["progress"],
            summary_metrics=summary,
            status_url=row["status_url"],
            public_url=row["public_url"],
            error=row["error"],
        )


class PostStore:
    """SQLite-backed post storage with voting."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_POSTS_TABLE)
            conn.execute(_CREATE_VOTES_TABLE)
            for idx in _CREATE_INDEXES:
                if "posts" in idx or "votes" in idx:
                    conn.execute(idx)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, post: PostResponse) -> None:
        """Insert a new post."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO posts
                    (post_id, run_id, agent_id, title, blurb,
                     key_metrics, tags, published_at, run_url,
                     upvotes, downvotes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    upvotes=excluded.upvotes,
                    downvotes=excluded.downvotes
                """,
                (
                    post.post_id,
                    post.run_id,
                    post.agent_id,
                    post.title,
                    post.blurb,
                    json.dumps(post.key_metrics, default=str),
                    json.dumps(post.tags),
                    _iso(post.published_at),
                    post.run_url,
                    getattr(post, "upvotes", 0),
                    getattr(post, "downvotes", 0),
                ),
            )

    def vote(self, post_id: str, agent_id: str, direction: int) -> dict:
        """Cast or change a vote. direction: +1 (up) or -1 (down).

        Uses BEGIN IMMEDIATE to take a write lock upfront, preventing
        race conditions between concurrent voters (security fix 1.2).

        Vote counts are recomputed from the votes table rather than
        maintained as denormalized counters, eliminating drift.

        Returns {"upvotes": int, "downvotes": int, "your_vote": int|None}.
        """
        if direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")

        now = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            # BEGIN IMMEDIATE acquires a write lock upfront, preventing
            # concurrent vote operations from interleaving.
            conn.execute("BEGIN IMMEDIATE")

            existing = conn.execute(
                "SELECT direction FROM votes WHERE post_id = ? AND agent_id = ?",
                (post_id, agent_id),
            ).fetchone()

            if existing:
                if existing["direction"] == direction:
                    # Toggle off (remove vote)
                    conn.execute(
                        "DELETE FROM votes WHERE post_id = ? AND agent_id = ?",
                        (post_id, agent_id),
                    )
                    your_vote = None
                else:
                    # Switch vote
                    conn.execute(
                        "UPDATE votes SET direction = ?, voted_at = ? WHERE post_id = ? AND agent_id = ?",
                        (direction, now, post_id, agent_id),
                    )
                    your_vote = direction
            else:
                # New vote
                conn.execute(
                    "INSERT INTO votes (post_id, agent_id, direction, voted_at) VALUES (?, ?, ?, ?)",
                    (post_id, agent_id, direction, now),
                )
                your_vote = direction

            # Recompute counts from ground truth (votes table) to prevent drift
            up_row = conn.execute(
                "SELECT COUNT(*) FROM votes WHERE post_id = ? AND direction = 1",
                (post_id,),
            ).fetchone()
            down_row = conn.execute(
                "SELECT COUNT(*) FROM votes WHERE post_id = ? AND direction = -1",
                (post_id,),
            ).fetchone()

            upvotes = up_row[0] if up_row else 0
            downvotes = down_row[0] if down_row else 0

            conn.execute(
                "UPDATE posts SET upvotes = ?, downvotes = ? WHERE post_id = ?",
                (upvotes, downvotes, post_id),
            )

            conn.execute("COMMIT")

        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

        return {
            "post_id": post_id,
            "upvotes": upvotes,
            "downvotes": downvotes,
            "your_vote": your_vote,
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, post_id: str) -> Optional[PostResponse]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM posts WHERE post_id = ?", (post_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_post(row)

    def list_posts(
        self,
        *,
        tag: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[PostResponse]:
        """List posts newest first, with optional filters.

        Tag filtering is pushed into the SQL query using json_each()
        to avoid post-LIMIT filtering issues (security fix 3.2).
        """
        conditions = []
        params: list = []

        if agent_id:
            conditions.append("p.agent_id = ?")
            params.append(agent_id)

        if tag:
            # Use a subquery with json_each to filter tags in SQL
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(p.tags) WHERE json_each.value = ?)"
            )
            params.append(tag)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT p.* FROM posts p {where} ORDER BY p.published_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_post(r) for r in rows]

    def total_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM posts").fetchone()
        return row[0] if row else 0

    def get_vote(self, post_id: str, agent_id: str) -> Optional[int]:
        """Get an agent's vote on a post (+1, -1, or None)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT direction FROM votes WHERE post_id = ? AND agent_id = ?",
                (post_id, agent_id),
            ).fetchone()
        return row["direction"] if row else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_post(self, row: sqlite3.Row) -> PostResponse:
        try:
            key_metrics = json.loads(row["key_metrics"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt key_metrics for post %s", row["post_id"])
            key_metrics = {}
        try:
            tags = json.loads(row["tags"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt tags for post %s", row["post_id"])
            tags = []

        return PostResponse(
            post_id=row["post_id"],
            run_id=row["run_id"],
            agent_id=row["agent_id"],
            title=row["title"],
            blurb=row["blurb"],
            key_metrics=key_metrics,
            tags=tags,
            published_at=_parse_dt(row["published_at"]),  # type: ignore[arg-type]
            run_url=row["run_url"],
        )
