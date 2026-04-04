"""Session Transcript Intelligence — embedding, indexing, and semantic search.

Provides the core API for the insights system:
- init_db(): Create SQLite tables with sqlite-vec
- index_session(): Parse, chunk, embed, and store a session
- query(): Semantic search over all indexed chunks

Uses sqlite-vec for vector storage and fastembed for local embeddings.
Install with: pip install -e ".[insights]"

See planning/SPEC.md for full specification.
"""

import dataclasses
import json
import struct
import uuid
from datetime import datetime
from pathlib import Path

# Use pysqlite3 for extension loading support (stdlib sqlite3 on macOS
# is often built without SQLITE_ENABLE_LOAD_EXTENSION)
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3
from typing import Optional

from .chunkers import TurnChunk, SubagentChunk, chunk_turns, chunk_subagents
from .entities import extract_entities
from .paths import decode_project_path

DB_PATH = str(Path.home() / ".claude" / "insights.db")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# ── Lazy model loading ────────────────────────────────────────

_EMBED_MODEL = None


def _get_embed_model():
    """Load fastembed model once, reuse across calls."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from fastembed import TextEmbedding
        _EMBED_MODEL = TextEmbedding(EMBEDDING_MODEL)
    return _EMBED_MODEL


def _embed_texts(texts: list[str], model=None) -> list[list[float]]:
    """Embed a list of texts. Uses lazy-loaded model if none provided."""
    if model is None:
        model = _get_embed_model()
    return [[float(x) for x in vec] for vec in model.embed(texts)]


def _serialize_vector(vec: list[float]) -> bytes:
    """Pack float32 vector into bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


# ── Database ──────────────────────────────────────────────────

@dataclasses.dataclass
class ChunkResult:
    """A search result from the insights database."""
    id: str
    session_id: str
    project_path: str
    chunk_type: str
    timestamp: str
    content: str
    metadata: dict
    distance: float


def get_db(path: str = DB_PATH) -> sqlite3.Connection:
    """Open insights DB with sqlite-vec extension loaded."""
    import sqlite_vec

    conn = sqlite3.connect(path, timeout=10)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # WAL mode allows concurrent readers + one writer.
    # May fail if another process holds an exclusive lock — that's OK,
    # it only needs to succeed once and the mode persists.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def init_db(conn: sqlite3.Connection):
    """Create tables and indexes if they don't exist."""
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            project_path TEXT NOT NULL,
            chunk_type TEXT NOT NULL
                CHECK(chunk_type IN ('turn', 'subagent_summary', 'chat')),
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_uuid TEXT,
            timestamp TEXT NOT NULL,
            indexed_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
        CREATE INDEX IF NOT EXISTS idx_chunks_ts ON chunks(timestamp);

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[{EMBEDDING_DIM}]
        );

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            value TEXT NOT NULL,
            UNIQUE(chunk_id, entity_type, value)
        );

        CREATE INDEX IF NOT EXISTS idx_entities_type_value
            ON entities(entity_type, value);
        CREATE INDEX IF NOT EXISTS idx_entities_session
            ON entities(session_id);

        CREATE TABLE IF NOT EXISTS insights_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        -- FTS5 full-text search (content-synced with chunks via rowid)
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(content, chunk_type, project_path, content=chunks, content_rowid=rowid);

        -- Auto-sync triggers for FTS
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content, chunk_type, project_path)
            VALUES (new.rowid, new.content, new.chunk_type, new.project_path);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content, chunk_type, project_path)
            VALUES ('delete', old.rowid, old.content, old.chunk_type, old.project_path);
        END;

        -- Playbooks — structured templates for recurring deliverables
        CREATE TABLE IF NOT EXISTS playbooks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            sections TEXT NOT NULL,
            data_steps TEXT NOT NULL,
            keywords TEXT NOT NULL,
            example_output TEXT NOT NULL DEFAULT '',
            intent_patterns TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            deleted_at TEXT
        );

        -- Chat conversation logs (persistent, searchable)
        CREATE TABLE IF NOT EXISTS chat_logs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            messages TEXT NOT NULL,
            created_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        -- Projects — aggregate sessions by project_path
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            session_count INTEGER NOT NULL DEFAULT 0,
            last_activity TEXT,
            created_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(path);

        -- Hierarchical summaries (L1=session, L2=project, L3=portfolio)
        CREATE TABLE IF NOT EXISTS summary_levels (
            id TEXT PRIMARY KEY,
            level INTEGER NOT NULL
                CHECK(level IN (1, 2, 3)),
            entity_id TEXT NOT NULL,
            entity_type TEXT NOT NULL
                CHECK(entity_type IN ('session', 'project', 'portfolio')),
            title TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            source_ids TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_summary_entity
            ON summary_levels(entity_type, entity_id);
        CREATE INDEX IF NOT EXISTS idx_summary_level
            ON summary_levels(level);
    """)
    conn.commit()
    _migrate_chunk_type_check(conn)
    _migrate_playbooks_schema(conn)
    _ensure_model_provenance(conn)
    _seed_playbooks(conn)


def _migrate_chunk_type_check(conn: sqlite3.Connection):
    """Expand chunk_type CHECK to include 'chat' for existing databases.

    Uses PRAGMA writable_schema to edit the CHECK constraint in-place,
    avoiding a table rebuild (which would break FTS triggers and vec_chunks rowids).
    """
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if not row or "'chat'" in row[0]:
            return  # Fresh DB or already migrated
        new_ddl = row[0].replace(
            "CHECK(chunk_type IN ('turn', 'subagent_summary'))",
            "CHECK(chunk_type IN ('turn', 'subagent_summary', 'chat'))",
        )
        if new_ddl == row[0]:
            return
        conn.execute("PRAGMA writable_schema = ON")
        conn.execute(
            "UPDATE sqlite_master SET sql = ? WHERE type='table' AND name='chunks'",
            (new_ddl,),
        )
        conn.execute("PRAGMA writable_schema = OFF")
        conn.commit()
    except Exception:
        pass  # Non-critical — worst case, 'chat' inserts will fail on old DBs


def _migrate_playbooks_schema(conn: sqlite3.Connection):
    """Add columns to playbooks table that may be missing from older DBs."""
    try:
        # Check if example_output column exists
        cols = [row[1] for row in conn.execute("PRAGMA table_info(playbooks)").fetchall()]
        if "example_output" not in cols:
            conn.execute("ALTER TABLE playbooks ADD COLUMN example_output TEXT NOT NULL DEFAULT ''")
            conn.commit()
        if "intent_patterns" not in cols:
            conn.execute("ALTER TABLE playbooks ADD COLUMN intent_patterns TEXT NOT NULL DEFAULT '[]'")
            conn.commit()
    except Exception:
        pass  # Table may not exist yet (first init)


def _seed_playbooks(conn: sqlite3.Connection):
    """Seed default playbooks if the table is empty."""
    from .playbooks import seed_defaults
    seed_defaults(conn)


def _ensure_model_provenance(conn: sqlite3.Connection):
    """Record current embedding model info. Called on every init_db()."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "model_name": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "updated_at": now,
    }
    for key, value in meta.items():
        conn.execute(
            """INSERT INTO insights_meta (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value,
               updated_at = excluded.updated_at""",
            (key, str(value), now),
        )
    conn.commit()


# ── Session deduplication ─────────────────────────────────────

def _session_indexed(conn: sqlite3.Connection, session_id: str) -> Optional[str]:
    """Check if session is already indexed. Returns indexed_at or None."""
    row = conn.execute(
        "SELECT indexed_at FROM chunks WHERE session_id = ? LIMIT 1",
        (session_id,),
    ).fetchone()
    return row[0] if row else None


def is_indexed(conn: sqlite3.Connection, session_id: str) -> bool:
    """Check if a session has been indexed into insights. Public API."""
    return _session_indexed(conn, session_id) is not None


def _delete_session_chunks(conn: sqlite3.Connection, session_id: str):
    """Remove all chunks, vectors, and entities for a session."""
    rowids = conn.execute(
        "SELECT rowid FROM chunks WHERE session_id = ?",
        (session_id,),
    ).fetchall()

    if rowids:
        placeholders = ",".join("?" * len(rowids))
        ids = [r[0] for r in rowids]
        conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM chunks WHERE rowid IN ({placeholders})", ids)

    # Clean up entities for this session
    conn.execute("DELETE FROM entities WHERE session_id = ?", (session_id,))


def _should_reindex(conn: sqlite3.Connection, session_id: str, session_path: Path) -> bool:
    """Check if a session needs (re)indexing based on file mtime vs indexed_at."""
    indexed_at = _session_indexed(conn, session_id)
    if indexed_at is None:
        return True

    try:
        file_mtime = session_path.stat().st_mtime
        indexed_dt = datetime.fromisoformat(indexed_at.replace("Z", "+00:00"))
        return file_mtime > indexed_dt.timestamp()
    except (OSError, ValueError):
        return True


# ── Indexing ──────────────────────────────────────────────────

def index_session(
    session_path: str,
    conn: sqlite3.Connection,
    model=None,
    session_id: str | None = None,
    project_path: str | None = None,
    summarize_subagents: bool = True,
    git_anchor: dict | None = None,
) -> tuple[int, int]:
    """Parse, chunk, embed, and store a session.

    Returns (turns_indexed, subagents_indexed).

    Args:
        session_path: Path to the .jsonl session file
        conn: SQLite connection with sqlite-vec loaded
        model: fastembed TextEmbedding model (lazy-loaded if None)
        session_id: Override session ID (defaults to filename stem)
        project_path: Override project path (defaults to decoded parent dir name)
        summarize_subagents: Ignored (kept for backward compat)
        git_anchor: Git context dict from get_git_anchor() — stored as entities
    """
    session_path = Path(session_path)

    if session_id is None:
        session_id = session_path.stem
    if project_path is None:
        project_path = decode_project_path(session_path.parent.name)

    # Check dedup
    if not _should_reindex(conn, session_id, session_path):
        return (0, 0)

    # If re-indexing, clear old data
    _delete_session_chunks(conn, session_id)

    # Chunk
    turn_chunks = chunk_turns(session_path)
    subagent_chunks = chunk_subagents(session_path, summarize=summarize_subagents)

    if not turn_chunks and not subagent_chunks:
        return (0, 0)

    # Embed all chunks in one batch
    all_texts = [c.content for c in turn_chunks] + [c.content for c in subagent_chunks]
    if model is None:
        model = _get_embed_model()
    embeddings = _embed_texts(all_texts, model)

    # Store in a single transaction
    turn_count = 0
    subagent_count = 0

    with conn:
        for i, chunk in enumerate(turn_chunks):
            chunk_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO chunks (id, session_id, project_path, chunk_type,
                   content, metadata, source_uuid, timestamp)
                   VALUES (?, ?, ?, 'turn', ?, ?, ?, ?)""",
                (chunk_id, session_id, project_path,
                 chunk.content, json.dumps(chunk.metadata),
                 chunk.user_uuid, chunk.timestamp),
            )
            rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_vector(embeddings[i])),
            )
            # Extract and store entities
            for etype, evalue in extract_entities(chunk.content, chunk.metadata):
                conn.execute(
                    """INSERT OR IGNORE INTO entities
                       (chunk_id, session_id, entity_type, value)
                       VALUES (?, ?, ?, ?)""",
                    (chunk_id, session_id, etype, evalue),
                )
            turn_count += 1

        offset = len(turn_chunks)
        for j, chunk in enumerate(subagent_chunks):
            chunk_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO chunks (id, session_id, project_path, chunk_type,
                   content, metadata, source_uuid, timestamp)
                   VALUES (?, ?, ?, 'subagent_summary', ?, ?, ?, ?)""",
                (chunk_id, session_id, project_path,
                 chunk.content, json.dumps(chunk.metadata),
                 chunk.slug, chunk.timestamp),
            )
            rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_vector(embeddings[offset + j])),
            )
            # Extract and store entities
            for etype, evalue in extract_entities(chunk.content, chunk.metadata):
                conn.execute(
                    """INSERT OR IGNORE INTO entities
                       (chunk_id, session_id, entity_type, value)
                       VALUES (?, ?, ?, ?)""",
                    (chunk_id, session_id, etype, evalue),
                )
            subagent_count += 1

        # Store git anchor entities (applied to first chunk as anchor point)
        if git_anchor and turn_chunks:
            first_chunk_id = conn.execute(
                "SELECT id FROM chunks WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            if first_chunk_id:
                anchor_chunk_id = first_chunk_id[0]
                if git_anchor.get("branch"):
                    conn.execute(
                        """INSERT OR IGNORE INTO entities
                           (chunk_id, session_id, entity_type, value)
                           VALUES (?, ?, 'git_branch', ?)""",
                        (anchor_chunk_id, session_id, git_anchor["branch"]),
                    )
                for h in git_anchor.get("commit_hashes", []):
                    conn.execute(
                        """INSERT OR IGNORE INTO entities
                           (chunk_id, session_id, entity_type, value)
                           VALUES (?, ?, 'git_commit', ?)""",
                        (anchor_chunk_id, session_id, h),
                    )
                for f in git_anchor.get("files_changed", []):
                    conn.execute(
                        """INSERT OR IGNORE INTO entities
                           (chunk_id, session_id, entity_type, value)
                           VALUES (?, ?, 'file_path', ?)""",
                        (anchor_chunk_id, session_id, f),
                    )

    # Update project stats after indexing
    if turn_count or subagent_count:
        try:
            update_project_stats(conn, project_path)
        except Exception:
            pass  # Non-critical — backfill_projects can catch up

    return (turn_count, subagent_count)


# ── Query ─────────────────────────────────────────────────────

def query(
    text: str,
    conn: sqlite3.Connection,
    model=None,
    limit: int = 10,
    chunk_type: str | None = None,
) -> list[ChunkResult]:
    """Semantic search over all indexed chunks.

    Returns results sorted by distance (ascending = most similar).

    Args:
        text: Natural language query
        conn: SQLite connection with sqlite-vec loaded
        model: fastembed TextEmbedding model (lazy-loaded if None)
        limit: Max results to return
        chunk_type: Optional filter ('turn' or 'subagent_summary')
    """
    if model is None:
        model = _get_embed_model()

    # Embed query
    query_vec = _embed_texts([text], model)[0]
    query_bytes = _serialize_vector(query_vec)

    # Vector search
    rows = conn.execute(
        """SELECT v.rowid, v.distance
           FROM vec_chunks v
           WHERE v.embedding MATCH ?
           ORDER BY v.distance
           LIMIT ?""",
        (query_bytes, limit * 2 if chunk_type else limit),
    ).fetchall()

    if not rows:
        return []

    # Fetch chunk metadata
    rowids = [r[0] for r in rows]
    distances = {r[0]: r[1] for r in rows}

    placeholders = ",".join("?" * len(rowids))
    type_filter = f"AND chunk_type = '{chunk_type}'" if chunk_type else ""
    chunks = conn.execute(
        f"""SELECT rowid, id, session_id, project_path, chunk_type,
                   timestamp, content, metadata
            FROM chunks
            WHERE rowid IN ({placeholders}) {type_filter}""",
        rowids,
    ).fetchall()

    # Assemble results
    results = []
    for row in chunks:
        results.append(ChunkResult(
            id=row[1],
            session_id=row[2],
            project_path=row[3],
            chunk_type=row[4],
            timestamp=row[5],
            content=row[6],
            metadata=json.loads(row[7]),
            distance=distances[row[0]],
        ))

    results.sort(key=lambda r: r.distance)
    return results[:limit]


# ── Entity query ──────────────────────────────────────────────

def query_by_entity(
    conn: sqlite3.Connection,
    entity_type: str,
    value: str,
    limit: int = 20,
    exact: bool = False,
) -> list[ChunkResult]:
    """Find chunks by entity type and value.

    Args:
        entity_type: 'file_path', 'error_class', 'url', 'git_branch', 'git_commit'
        value: Entity value to search for
        limit: Max results
        exact: If True, exact match. If False, LIKE '%value%'
    """
    if exact:
        rows = conn.execute(
            """SELECT DISTINCT e.chunk_id
               FROM entities e
               WHERE e.entity_type = ? AND e.value = ?
               LIMIT ?""",
            (entity_type, value, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT DISTINCT e.chunk_id
               FROM entities e
               WHERE e.entity_type = ? AND e.value LIKE ?
               LIMIT ?""",
            (entity_type, f"%{value}%", limit),
        ).fetchall()

    if not rows:
        return []

    chunk_ids = [r[0] for r in rows]
    placeholders = ",".join("?" * len(chunk_ids))
    chunks = conn.execute(
        f"""SELECT id, session_id, project_path, chunk_type,
                   timestamp, content, metadata
            FROM chunks
            WHERE id IN ({placeholders})
            ORDER BY timestamp DESC""",
        chunk_ids,
    ).fetchall()

    return [
        ChunkResult(
            id=row[0], session_id=row[1], project_path=row[2],
            chunk_type=row[3], timestamp=row[4], content=row[5],
            metadata=json.loads(row[6]), distance=0.0,
        )
        for row in chunks
    ]


# ── FTS5 full-text search ─────────────────────────────────────

import re as _re

def _fts_search(
    text: str,
    conn: sqlite3.Connection,
    limit: int = 20,
) -> list[ChunkResult]:
    """Full-text search using FTS5.

    Tokenizes query into AND-joined terms for FTS5 MATCH.
    Returns results with distance derived from FTS5 rank.
    """
    # Tokenize: keep alphanumeric words 2+ chars
    words = _re.findall(r'\w{2,}', text.lower())
    if not words:
        return []

    # FTS5 query: join with AND
    fts_query = " AND ".join(f'"{w}"' for w in words)

    try:
        rows = conn.execute(
            """SELECT f.rowid, f.rank
               FROM chunks_fts f
               WHERE chunks_fts MATCH ?
               ORDER BY f.rank
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
    except Exception:
        # FTS table may not exist on older DBs
        return []

    if not rows:
        return []

    # Fetch full chunk data
    rowids = [r[0] for r in rows]
    ranks = {r[0]: r[1] for r in rows}
    placeholders = ",".join("?" * len(rowids))

    chunks = conn.execute(
        f"""SELECT rowid, id, session_id, project_path, chunk_type,
                   timestamp, content, metadata
            FROM chunks
            WHERE rowid IN ({placeholders})""",
        rowids,
    ).fetchall()

    results = []
    for row in chunks:
        results.append(ChunkResult(
            id=row[1],
            session_id=row[2],
            project_path=row[3],
            chunk_type=row[4],
            timestamp=row[5],
            content=row[6],
            metadata=json.loads(row[7]),
            distance=-ranks[row[0]],  # FTS5 rank is negative; invert so lower = better
        ))

    results.sort(key=lambda r: r.distance)
    return results


def _auto_entity_search(
    text: str,
    conn: sqlite3.Connection,
    limit: int = 10,
) -> list[ChunkResult]:
    """Auto-detect entities in query text and search for them.

    Detects: file paths, error classes, git branches.
    """
    from .entities import _FILE_PATH_RE, _RELATIVE_PATH_RE, _ERROR_CLASS_RE

    detected: list[tuple[str, str]] = []

    # File paths (absolute)
    for m in _FILE_PATH_RE.finditer(text):
        detected.append(("file_path", m.group(1)))
    # File paths (relative)
    for m in _RELATIVE_PATH_RE.finditer(text):
        detected.append(("file_path", m.group(1)))
    # Error classes
    for m in _ERROR_CLASS_RE.finditer(text):
        detected.append(("error_class", m.group(1)))
    # Git branches: feature/xxx, fix/xxx, etc.
    for m in _re.finditer(r'\b((?:feature|fix|hotfix|release|bugfix)/[\w._-]+)\b', text):
        detected.append(("git_branch", m.group(1)))

    if not detected:
        return []

    seen_ids: set[str] = set()
    results: list[ChunkResult] = []

    for entity_type, value in detected:
        for r in query_by_entity(conn, entity_type, value, limit=limit):
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                results.append(r)

    return results[:limit]


def rrf_search(
    text: str,
    conn: sqlite3.Connection,
    model=None,
    limit: int = 10,
    k: int = 60,
) -> list[ChunkResult]:
    """Reciprocal Rank Fusion combining semantic, FTS5, and entity search.

    RRF formula: score(d) = Σ 1/(k + rank_i(d))
    Returns top `limit` results with distance = 1 - rrf_score.
    """
    if model is None:
        model = _get_embed_model()

    # Run all three retrievers
    results_semantic = query(text, conn, model, limit=20)
    results_fts = _fts_search(text, conn, limit=20)
    results_entity = _auto_entity_search(text, conn, limit=10)

    # Build RRF scores keyed by chunk id
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, ChunkResult] = {}

    for result_list in [results_semantic, results_fts, results_entity]:
        for rank, r in enumerate(result_list, start=1):
            rrf_scores[r.id] = rrf_scores.get(r.id, 0.0) + 1.0 / (k + rank)
            if r.id not in chunk_map:
                chunk_map[r.id] = r

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    results = []
    for cid in sorted_ids[:limit]:
        r = chunk_map[cid]
        results.append(ChunkResult(
            id=r.id,
            session_id=r.session_id,
            project_path=r.project_path,
            chunk_type=r.chunk_type,
            timestamp=r.timestamp,
            content=r.content,
            metadata=r.metadata,
            distance=1.0 - rrf_scores[cid],  # lower = better, consistent with existing API
        ))

    return results


# ── Stats ─────────────────────────────────────────────────────

def get_stats(conn: sqlite3.Connection) -> dict:
    """Get index statistics including model provenance."""
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    turns = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_type = 'turn'"
    ).fetchone()[0]
    subagents = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_type = 'subagent_summary'"
    ).fetchone()[0]
    sessions = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM chunks"
    ).fetchone()[0]

    # Model provenance from insights_meta
    meta = {}
    try:
        rows = conn.execute("SELECT key, value FROM insights_meta").fetchall()
        meta = {r[0]: r[1] for r in rows}
    except Exception:
        pass

    # Entity counts
    total_entities = 0
    entity_breakdown = {}
    try:
        total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        for row in conn.execute(
            "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type"
        ).fetchall():
            entity_breakdown[row[0]] = row[1]
    except Exception:
        pass

    return {
        "total_chunks": total,
        "turns": turns,
        "subagent_summaries": subagents,
        "sessions_indexed": sessions,
        "total_entities": total_entities,
        "entities_by_type": entity_breakdown,
        "model_name": meta.get("model_name", "unknown"),
        "embedding_dim": meta.get("embedding_dim", "unknown"),
    }


# ── Projects ─────────────────────────────────────────────────

def backfill_projects(conn: sqlite3.Connection) -> int:
    """Populate projects table from distinct project_paths in chunks.

    Inserts new projects and updates session_count/last_activity for
    existing ones. Returns number of projects upserted.
    """
    rows = conn.execute(
        """SELECT project_path,
                  COUNT(DISTINCT session_id) AS session_count,
                  MAX(timestamp) AS last_activity
           FROM chunks
           GROUP BY project_path"""
    ).fetchall()

    count = 0
    now = datetime.now().isoformat()
    for path, session_count, last_activity in rows:
        # Derive name from last path component
        name = Path(path).name if path else "unknown"
        project_id = str(uuid.uuid5(uuid.NAMESPACE_URL, path))
        conn.execute(
            """INSERT INTO projects (id, path, name, session_count, last_activity, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   session_count = excluded.session_count,
                   last_activity = excluded.last_activity,
                   updated_at = excluded.updated_at""",
            (project_id, path, name, session_count, last_activity, now),
        )
        count += 1
    conn.commit()
    return count


def update_project_stats(conn: sqlite3.Connection, project_path: str):
    """Update a single project's session_count and last_activity after indexing."""
    row = conn.execute(
        """SELECT COUNT(DISTINCT session_id), MAX(timestamp)
           FROM chunks WHERE project_path = ?""",
        (project_path,),
    ).fetchone()
    if not row or not row[0]:
        return

    session_count, last_activity = row
    now = datetime.now().isoformat()
    name = Path(project_path).name if project_path else "unknown"
    project_id = str(uuid.uuid5(uuid.NAMESPACE_URL, project_path))
    conn.execute(
        """INSERT INTO projects (id, path, name, session_count, last_activity, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(path) DO UPDATE SET
               session_count = excluded.session_count,
               last_activity = excluded.last_activity,
               updated_at = excluded.updated_at""",
        (project_id, project_path, name, session_count, last_activity, now),
    )
    conn.commit()


def list_projects(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """List all projects, ordered by last_activity descending."""
    rows = conn.execute(
        """SELECT id, path, name, session_count, last_activity, created_at, updated_at
           FROM projects
           ORDER BY last_activity DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {
            "id": r[0], "path": r[1], "name": r[2],
            "session_count": r[3], "last_activity": r[4],
            "created_at": r[5], "updated_at": r[6],
        }
        for r in rows
    ]


def get_project(conn: sqlite3.Connection, project_path: str) -> dict | None:
    """Get a single project by path."""
    row = conn.execute(
        "SELECT id, path, name, session_count, last_activity FROM projects WHERE path = ?",
        (project_path,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0], "path": row[1], "name": row[2],
        "session_count": row[3], "last_activity": row[4],
    }


# ── Summary Levels ───────────────────────────────────────────

def get_summary(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    level: int | None = None,
) -> dict | None:
    """Get the most recent summary for an entity."""
    if level is not None:
        row = conn.execute(
            """SELECT id, level, entity_id, entity_type, title, summary_text,
                      source_ids, created_at, updated_at
               FROM summary_levels
               WHERE entity_type = ? AND entity_id = ? AND level = ?
               ORDER BY updated_at DESC LIMIT 1""",
            (entity_type, entity_id, level),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT id, level, entity_id, entity_type, title, summary_text,
                      source_ids, created_at, updated_at
               FROM summary_levels
               WHERE entity_type = ? AND entity_id = ?
               ORDER BY level DESC, updated_at DESC LIMIT 1""",
            (entity_type, entity_id),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row[0], "level": row[1], "entity_id": row[2],
        "entity_type": row[3], "title": row[4], "summary_text": row[5],
        "source_ids": json.loads(row[6]), "created_at": row[7],
        "updated_at": row[8],
    }


def upsert_summary(
    conn: sqlite3.Connection,
    level: int,
    entity_id: str,
    entity_type: str,
    title: str,
    summary_text: str,
    source_ids: list[str] | None = None,
) -> str:
    """Insert or update a summary. Returns the summary ID."""
    now = datetime.now().isoformat()
    sources_json = json.dumps(source_ids or [])

    # Check for existing summary at this level+entity
    existing = conn.execute(
        """SELECT id FROM summary_levels
           WHERE level = ? AND entity_id = ? AND entity_type = ?""",
        (level, entity_id, entity_type),
    ).fetchone()

    if existing:
        summary_id = existing[0]
        conn.execute(
            """UPDATE summary_levels
               SET title = ?, summary_text = ?, source_ids = ?, updated_at = ?
               WHERE id = ?""",
            (title, summary_text, sources_json, now, summary_id),
        )
    else:
        summary_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO summary_levels
               (id, level, entity_id, entity_type, title, summary_text, source_ids, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (summary_id, level, entity_id, entity_type, title, summary_text, sources_json, now),
        )
    conn.commit()
    return summary_id


def get_project_summary(conn: sqlite3.Connection, project_path: str) -> dict | None:
    """Convenience: get the L2 project summary for a project path."""
    return get_summary(conn, "project", project_path, level=2)


def get_portfolio_summary(conn: sqlite3.Connection) -> dict | None:
    """Convenience: get the L3 portfolio rollup."""
    return get_summary(conn, "portfolio", "global", level=3)


def list_stale_projects(
    conn: sqlite3.Connection,
    limit: int = 20,
) -> list[dict]:
    """Find projects whose L2 summary is older than their latest session activity.

    Returns projects that need L2 regeneration.
    """
    rows = conn.execute(
        """SELECT p.id, p.path, p.name, p.session_count, p.last_activity,
                  s.updated_at AS summary_updated
           FROM projects p
           LEFT JOIN summary_levels s
               ON s.entity_id = p.path AND s.entity_type = 'project' AND s.level = 2
           WHERE s.id IS NULL
              OR s.updated_at < p.last_activity
           ORDER BY p.last_activity DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {
            "id": r[0], "path": r[1], "name": r[2],
            "session_count": r[3], "last_activity": r[4],
            "summary_updated": r[5],
        }
        for r in rows
    ]
