"""Tests for reindex module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

try:
    import sqlite_vec
    from fastembed import TextEmbedding
    HAS_INSIGHTS = True
except ImportError:
    HAS_INSIGHTS = False

pytestmark = pytest.mark.skipif(not HAS_INSIGHTS, reason="insights deps not installed")


@pytest.fixture
def indexed_db(tmp_path):
    """Create a temporary insights DB with one indexed session."""
    from claude_session_commons.insights import get_db, init_db, index_session

    # Create a fixture session file
    session_file = tmp_path / "projects" / "testproj" / "test-session-001.jsonl"
    session_file.parent.mkdir(parents=True)

    entries = [
        {"type": "user", "uuid": "u1", "timestamp": "2026-02-20T10:00:00Z",
         "message": {"content": "Create a user authentication system"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "message": {"model": "claude", "content": [
             {"type": "text", "text": "I'll create an auth system with JWT tokens and password hashing."}
         ]}},
        {"type": "user", "uuid": "u2", "timestamp": "2026-02-20T10:05:00Z",
         "message": {"content": "Add login and logout endpoints"}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "message": {"model": "claude", "content": [
             {"type": "text", "text": "Added POST /login and POST /logout endpoints with session management."}
         ]}},
    ]
    session_file.write_text("\n".join(json.dumps(e) for e in entries))

    db_path = str(tmp_path / "insights.db")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    conn = get_db(db_path)
    init_db(conn)

    index_session(str(session_file), conn, model=model,
                  session_id="test-session-001",
                  project_path="/Users/dev/testproj")

    yield conn, tmp_path, db_path, model, session_file
    conn.close()


def test_reindex_dry_run(indexed_db):
    """Dry run reports what would happen without changes."""
    conn, tmp_path, db_path, model, session_file = indexed_db
    conn.close()

    from claude_session_commons.reindex import reindex

    with patch("claude_session_commons.reindex.PROJECTS_DIR", tmp_path / "projects"), \
         patch("claude_session_commons.insights.DB_PATH", db_path):
        result = reindex(force=True, dry_run=True, verbose=False)

    assert result["reindexed"] >= 1
    assert result["errors"] == 0


def test_reindex_force(indexed_db):
    """Force reindex re-embeds all sessions."""
    conn, tmp_path, db_path, model, session_file = indexed_db

    initial_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert initial_count > 0
    conn.close()

    from claude_session_commons.reindex import reindex

    with patch("claude_session_commons.reindex.PROJECTS_DIR", tmp_path / "projects"), \
         patch("claude_session_commons.insights.DB_PATH", db_path):
        result = reindex(force=True, dry_run=False, verbose=False)

    assert result["reindexed"] >= 1
    assert result["errors"] == 0


def test_reindex_orphan_removal(indexed_db):
    """Orphaned sessions (file deleted) are cleaned up."""
    conn, tmp_path, db_path, model, session_file = indexed_db

    # Delete the session file to make it orphaned
    session_file.unlink()
    conn.close()

    from claude_session_commons.reindex import reindex

    with patch("claude_session_commons.reindex.PROJECTS_DIR", tmp_path / "projects"), \
         patch("claude_session_commons.insights.DB_PATH", db_path):
        result = reindex(force=False, dry_run=False, verbose=False)

    assert result["orphans_removed"] >= 1
