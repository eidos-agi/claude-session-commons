"""Tests for insights.py — indexing and semantic search.

These tests use sqlite-vec and fastembed. Skip if not installed:
    pip install -e ".[insights]"
"""

import json
import sqlite3
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

from claude_session_commons.insights import (
    get_db,
    init_db,
    index_session,
    query,
    get_stats,
    ChunkResult,
    _serialize_vector,
    _embed_texts,
    _fts_search,
    _auto_entity_search,
    rrf_search,
    EMBEDDING_DIM,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_session.jsonl"


@pytest.fixture
def db(tmp_path):
    """Create a temporary insights database."""
    db_path = str(tmp_path / "test_insights.db")
    conn = get_db(db_path)
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def embed_model():
    """Load embedding model once for all tests in this module."""
    return TextEmbedding("BAAI/bge-small-en-v1.5")


# ── Database setup tests ─────────────────────────────────────

def test_init_db_creates_tables(db):
    """init_db should create chunks and vec_chunks tables."""
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "chunks" in table_names
    assert "vec_chunks" in table_names


def test_init_db_idempotent(db):
    """Calling init_db twice should not raise."""
    init_db(db)  # second call
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    assert len([t for t in tables if t[0] == "chunks"]) == 1


# ── Embedding tests ──────────────────────────────────────────

def test_embed_texts(embed_model):
    """fastembed should produce 384-dim vectors."""
    vecs = _embed_texts(["hello world", "test query"], embed_model)
    assert len(vecs) == 2
    assert len(vecs[0]) == EMBEDDING_DIM
    assert all(isinstance(v, (float, int)) or hasattr(v, '__float__') for v in vecs[0])


def test_serialize_vector():
    """Vector serialization should produce correct byte count."""
    vec = [0.1] * EMBEDDING_DIM
    data = _serialize_vector(vec)
    assert len(data) == EMBEDDING_DIM * 4  # float32 = 4 bytes


# ── Indexing tests ────────────────────────────────────────────

def test_index_session_basic(db, embed_model):
    """Index fixture session and verify chunks were stored."""
    turns, subagents = index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )
    assert turns >= 4  # fixture has 6 user-assistant pairs
    # Subagent threshold is 10; fixture has only 6 progress entries per slug
    assert subagents == 0

    stats = get_stats(db)
    assert stats["total_chunks"] == turns
    assert stats["sessions_indexed"] == 1


def test_index_session_dedup(db, embed_model):
    """Re-indexing same session should not duplicate chunks."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )
    first_count = get_stats(db)["total_chunks"]

    # Index again — should skip (file hasn't changed)
    turns, subagents = index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )
    assert turns == 0 and subagents == 0
    assert get_stats(db)["total_chunks"] == first_count


def test_index_session_chunk_types(db, embed_model):
    """Verify both turn and subagent_summary chunk types are stored."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )
    turn_count = db.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_type = 'turn'"
    ).fetchone()[0]
    sub_count = db.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_type = 'subagent_summary'"
    ).fetchone()[0]
    assert turn_count >= 4
    # Subagent threshold is 10; fixture subagents are below threshold
    assert sub_count == 0


# ── Query tests ───────────────────────────────────────────────

def test_query_returns_results(db, embed_model):
    """Query should return relevant results."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = query("database schema for users and roles", db, model=embed_model)
    assert len(results) > 0
    assert all(isinstance(r, ChunkResult) for r in results)

    # Results should be sorted by distance
    for i in range(len(results) - 1):
        assert results[i].distance <= results[i + 1].distance


def test_query_relevance(db, embed_model):
    """Top result for a targeted query should be relevant."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = query("soft delete deleted_at column", db, model=embed_model, limit=3)
    assert len(results) > 0
    # The top result should mention "deleted_at" or "soft delete"
    top = results[0]
    assert "deleted_at" in top.content.lower() or "soft delete" in top.content.lower()


def test_query_type_filter(db, embed_model):
    """Query with type filter should only return matching chunks."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = query("RBAC best practices", db, model=embed_model, chunk_type="subagent_summary")
    for r in results:
        assert r.chunk_type == "subagent_summary"


def test_query_limit(db, embed_model):
    """Query should respect the limit parameter."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = query("database", db, model=embed_model, limit=2)
    assert len(results) <= 2


def test_query_empty_db(db, embed_model):
    """Query on empty database should return empty list."""
    results = query("anything", db, model=embed_model)
    assert results == []


def test_query_result_metadata(db, embed_model):
    """Result metadata should be properly deserialized."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = query("database schema", db, model=embed_model, limit=1)
    assert len(results) == 1
    r = results[0]
    assert r.session_id == "test-session-001"
    assert r.project_path == "/Users/dev/myproject"
    assert isinstance(r.metadata, dict)
    assert r.timestamp
    assert r.distance >= 0


# ── FTS5 tests ───────────────────────────────────────────────

def test_fts5_table_created(db):
    """init_db should create the chunks_fts FTS5 table."""
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "chunks_fts" in table_names


def test_fts_search_returns_results(db, embed_model):
    """FTS search should find chunks by keyword."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = _fts_search("database schema", db, limit=5)
    assert len(results) > 0
    assert all(isinstance(r, ChunkResult) for r in results)


def test_fts_search_empty_query(db):
    """FTS search with empty/nonsense query should return empty list."""
    results = _fts_search("", db)
    assert results == []


def test_fts_search_no_match(db, embed_model):
    """FTS search for nonexistent term should return empty list."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = _fts_search("xyzzyplughtwisty", db)
    assert results == []


# ── RRF tests ─────────────────────────────────────────────────

def test_rrf_search_returns_results(db, embed_model):
    """RRF search should return combined results."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = rrf_search("database schema", db, model=embed_model, limit=5)
    assert len(results) > 0
    assert all(isinstance(r, ChunkResult) for r in results)


def test_rrf_search_respects_limit(db, embed_model):
    """RRF search should respect the limit parameter."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = rrf_search("database", db, model=embed_model, limit=2)
    assert len(results) <= 2


def test_rrf_search_empty_db(db, embed_model):
    """RRF search on empty database should return empty list."""
    results = rrf_search("anything", db, model=embed_model)
    assert results == []


def test_rrf_search_distance_consistency(db, embed_model):
    """RRF results should have distance values where lower = better."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = rrf_search("database schema", db, model=embed_model, limit=5)
    assert len(results) > 1
    # Results should be sorted by distance ascending (lower = better)
    for i in range(len(results) - 1):
        assert results[i].distance <= results[i + 1].distance


def test_rrf_deduplicates_across_retrievers(db, embed_model):
    """RRF should not return duplicate chunks from different retrievers."""
    index_session(
        str(FIXTURE), db, model=embed_model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )

    results = rrf_search("database schema", db, model=embed_model, limit=10)
    ids = [r.id for r in results]
    assert len(ids) == len(set(ids)), "RRF returned duplicate chunk IDs"
