"""Persistent chat log storage and indexing.

Saves web chat conversations to the insights DB so they're searchable
via the same RRF pipeline as session transcripts. Each chat becomes
both a row in chat_logs (full history) and a chunk in the chunks table
(searchable via semantic + FTS).
"""

import json
import logging
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = str(Path.home() / ".claude" / "insights.db")


def _open_write_conn():
    """Open a dedicated write connection with long busy timeout."""
    from ..insights import get_db
    return get_db(DB_PATH)


def save_chat(conn, chat_id: str, messages: list[dict], model=None):
    """Persist a chat conversation and index it as a searchable chunk.

    Called after each completed agent turn. Upserts both chat_logs
    (full conversation) and chunks (searchable content).

    Uses its own connection with retries to handle daemon write contention.
    The passed conn is ignored for writes — we open a dedicated connection.
    """
    if not messages:
        return

    # Auto-title from first user message
    title = "Untitled chat"
    for msg in messages:
        if msg.get("role") == "user":
            text = msg["content"].strip()
            title = text[:80] + ("..." if len(text) > 80 else "")
            break

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    messages_json = json.dumps(messages)
    content = _build_chunk_content(title, messages)
    chunk_id = f"chat-{chat_id}"

    # Compute embedding before acquiring write lock
    vec_bytes = None
    if model is not None:
        try:
            vecs = list(model.embed([content]))
            if vecs:
                vec_bytes = struct.pack(f"{len(vecs[0])}f", *[float(x) for x in vecs[0]])
        except Exception:
            pass

    # Retry write with backoff — daemon may hold write lock for seconds at a time
    for attempt in range(5):
        try:
            wconn = _open_write_conn()
            try:
                _do_write(wconn, chat_id, title, messages_json, now,
                          chunk_id, content, vec_bytes)
                return  # Success
            finally:
                wconn.close()
        except Exception as e:
            if "locked" in str(e) and attempt < 4:
                time.sleep(1 + attempt)  # 1s, 2s, 3s, 4s backoff
                continue
            logger.warning("chat persist failed after %d attempts: %s", attempt + 1, e)
            return


def _do_write(conn, chat_id, title, messages_json, now, chunk_id, content, vec_bytes):
    """Execute all writes in a single short transaction."""
    # Upsert chat_logs
    conn.execute(
        """INSERT INTO chat_logs (id, title, messages, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               title = excluded.title,
               messages = excluded.messages,
               updated_at = excluded.updated_at""",
        (chat_id, title, messages_json, now, now),
    )

    # Delete old chunk + vector if updating
    old_rowid = conn.execute(
        "SELECT rowid FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    if old_rowid:
        conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (old_rowid[0],))
    conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

    # Insert chunk
    conn.execute(
        """INSERT INTO chunks (id, session_id, project_path, chunk_type, content,
           metadata, timestamp, indexed_at)
           VALUES (?, ?, ?, 'chat', ?, ?, ?, ?)""",
        (
            chunk_id,
            chat_id,
            "web-chat",
            content,
            json.dumps({"chat_id": chat_id, "title": title, "turns": len(messages_json)}),
            now,
            now,
        ),
    )

    # Insert vector if available
    if vec_bytes:
        rowid = conn.execute(
            "SELECT rowid FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
            (rowid, vec_bytes),
        )

    conn.commit()


def load_chat(conn, chat_id: str) -> dict | None:
    """Load a saved chat conversation. Returns dict with id, title, messages, timestamps."""
    row = conn.execute(
        "SELECT id, title, messages, created_at, updated_at FROM chat_logs WHERE id = ?",
        (chat_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "messages": json.loads(row[2]),
        "created_at": row[3],
        "updated_at": row[4],
    }


def list_recent_chats(conn, limit: int = 20) -> list[dict]:
    """List recent chat conversations, newest first."""
    rows = conn.execute(
        """SELECT id, title, created_at, updated_at
           FROM chat_logs ORDER BY updated_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
        for r in rows
    ]


def _build_chunk_content(title: str, messages: list[dict]) -> str:
    """Build a searchable text chunk from a chat conversation.

    Concatenates user and assistant messages, truncated to 2000 chars
    to match the chunking limits used for session turns.
    """
    parts = [f"CHAT: {title}\n"]
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")

    text = "\n".join(parts)
    if len(text) > 2000:
        text = text[:2000]
    return text
