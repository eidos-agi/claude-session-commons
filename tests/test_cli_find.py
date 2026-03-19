"""Smoke tests for cli_find.py — claude-find CLI."""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import sqlite_vec
    from fastembed import TextEmbedding
    HAS_INSIGHTS = True
except ImportError:
    HAS_INSIGHTS = False

pytestmark = pytest.mark.skipif(not HAS_INSIGHTS, reason="insights deps not installed")

from claude_session_commons.cli_find import main, _format_result
from claude_session_commons.insights import get_db, init_db, index_session, ChunkResult

FIXTURE = Path(__file__).parent / "fixtures" / "sample_session.jsonl"


@pytest.fixture
def populated_db(tmp_path):
    """Create and populate a temporary insights database."""
    db_path = str(tmp_path / "test_find.db")
    conn = get_db(db_path)
    init_db(conn)
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    index_session(
        str(FIXTURE), conn, model=model,
        session_id="test-session-001",
        project_path="/Users/dev/myproject",
        summarize_subagents=False,
    )
    yield db_path, conn
    conn.close()


def test_format_result():
    """_format_result should produce readable output."""
    r = ChunkResult(
        id="abc",
        session_id="sess-1",
        project_path="/Users/dev/myproject",
        chunk_type="turn",
        timestamp="2026-02-18T10:00:00Z",
        content="USER: Hello\n\nASSISTANT: Hi there",
        metadata={"model": "claude-opus-4-6"},
        distance=0.23,
    )
    output = _format_result(r)
    assert "[0.23]" in output
    assert "2026-02-18 10:00" in output
    assert "turn" in output
    assert "USER: Hello" in output


def test_cli_stats(populated_db, capsys):
    """--stats flag should print index statistics."""
    db_path, _ = populated_db
    with patch("sys.argv", ["claude-find", "--stats", "--db", db_path]):
        main()
    captured = capsys.readouterr()
    assert "Sessions indexed:" in captured.out
    assert "Total chunks:" in captured.out


def test_cli_query(populated_db, capsys):
    """Basic query should return results."""
    db_path, _ = populated_db
    with patch("sys.argv", ["claude-find", "database schema", "--db", db_path, "-n", "3"]):
        main()
    captured = capsys.readouterr()
    assert "USER:" in captured.out
    assert "turn" in captured.out or "subagent_summary" in captured.out


def test_cli_type_filter(populated_db, capsys):
    """--type flag should filter results."""
    db_path, _ = populated_db
    with patch("sys.argv", ["claude-find", "RBAC", "--db", db_path, "--type", "subagent_summary"]):
        try:
            main()
        except SystemExit:
            pass  # cli exits 0 when no results found
    captured = capsys.readouterr()
    # Should only show subagent results (or no results if query doesn't match)
    if "subagent_summary" in captured.out:
        assert "turn" not in captured.out.split("subagent_summary")[0]


def test_cli_no_query(capsys):
    """No query and no --stats should print help."""
    with patch("sys.argv", ["claude-find"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
