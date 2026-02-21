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

    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db(conn: sqlite3.Connection):
    """Create tables and indexes if they don't exist."""
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            project_path TEXT NOT NULL,
            chunk_type TEXT NOT NULL
                CHECK(chunk_type IN ('turn', 'subagent_summary')),
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

        CREATE TABLE IF NOT EXISTS insights_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
    """)
    conn.commit()
    _ensure_model_provenance(conn)


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


def _delete_session_chunks(conn: sqlite3.Connection, session_id: str):
    """Remove all chunks and vectors for a session."""
    rowids = conn.execute(
        "SELECT rowid FROM chunks WHERE session_id = ?",
        (session_id,),
    ).fetchall()

    if rowids:
        placeholders = ",".join("?" * len(rowids))
        ids = [r[0] for r in rowids]
        conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM chunks WHERE rowid IN ({placeholders})", ids)


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
) -> tuple[int, int]:
    """Parse, chunk, embed, and store a session.

    Returns (turns_indexed, subagents_indexed).

    Args:
        session_path: Path to the .jsonl session file
        conn: SQLite connection with sqlite-vec loaded
        model: fastembed TextEmbedding model (lazy-loaded if None)
        session_id: Override session ID (defaults to filename stem)
        project_path: Override project path (defaults to decoded parent dir name)
        summarize_subagents: Whether to call claude -p for subagent summaries
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
            subagent_count += 1

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

    return {
        "total_chunks": total,
        "turns": turns,
        "subagent_summaries": subagents,
        "sessions_indexed": sessions,
        "model_name": meta.get("model_name", "unknown"),
        "embedding_dim": meta.get("embedding_dim", "unknown"),
    }
