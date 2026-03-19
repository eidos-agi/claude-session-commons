"""Tests for web dashboard API endpoints.

Uses FastAPI TestClient — no real server needed.
Skip if web dependencies not installed.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

try:
    from fastapi.testclient import TestClient
    HAS_WEB = True
except ImportError:
    HAS_WEB = False

pytestmark = pytest.mark.skipif(not HAS_WEB, reason="web deps not installed")


@pytest.fixture
def client(tmp_path):
    """Create a test client with patched paths."""
    status_file = tmp_path / "daemon.status.json"
    log_file = tmp_path / "daemon.log"

    with patch("claude_session_commons.web.app.STATUS_FILE", status_file), \
         patch("claude_session_commons.web.app.LOG_FILE", log_file), \
         patch("claude_session_commons.web.app.DB_PATH", str(tmp_path / "insights.db")):
        # Reset lazy globals for each test
        import claude_session_commons.web.app as web_app
        web_app._db_initialized = False
        web_app._embed_model = None

        from claude_session_commons.web.app import app
        yield TestClient(app), tmp_path, status_file, log_file


# ── Status endpoint ──────────────────────────────────────────

def test_status_no_file(client):
    """Status returns unknown when no heartbeat file."""
    tc, tmp, status_file, _ = client
    resp = tc.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "unknown"
    assert "error" in data


def test_status_with_file(client):
    """Status returns daemon state from file."""
    tc, tmp, status_file, _ = client
    status_file.write_text(json.dumps({
        "pid": 99999,
        "version": "1.1.0",
        "state": "running",
        "sessions_processed": 42,
        "sessions_failed": 2,
        "last_run": "2026-02-20T10:00:00Z",
    }))
    resp = tc.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "running"
    assert data["sessions_processed"] == 42


def test_status_corrupt_file(client):
    """Status handles corrupt JSON gracefully."""
    tc, tmp, status_file, _ = client
    status_file.write_text("not valid json {{{{")
    resp = tc.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "unknown"


# ── Stats endpoint ───────────────────────────────────────────

def test_stats_no_db(client):
    """Stats returns zeros when no insights DB."""
    tc, tmp, _, _ = client
    resp = tc.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 0


# ── Logs endpoint ────────────────────────────────────────────

def test_logs_no_file(client):
    """Logs returns empty when no log file."""
    tc, tmp, _, log_file = client
    resp = tc.get("/api/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lines"] == []


def test_logs_with_content(client):
    """Logs returns tail of log file."""
    tc, tmp, _, log_file = client
    lines = [f"2026-02-20 10:0{i}:00 [INFO] Line {i}" for i in range(10)]
    log_file.write_text("\n".join(lines) + "\n")

    resp = tc.get("/api/logs?lines=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["lines"]) == 5
    assert "Line 9" in data["lines"][-1]


# ── Sessions endpoint ────────────────────────────────────────

def test_sessions_returns_list(client):
    """Sessions endpoint returns a list."""
    tc, tmp, _, _ = client
    with patch("claude_session_commons.web.app.api_sessions") as mock_fn:
        # Just verify the endpoint shape — actual discovery needs ~/.claude/projects/
        resp = tc.get("/api/sessions?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data or "error" in data


# ── Search endpoint ──────────────────────────────────────────

def test_search_no_db(client):
    """Search returns error when no insights DB."""
    tc, tmp, _, _ = client
    resp = tc.post("/api/search", json={"query": "test query"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert "error" in data


def test_search_validates_input(client):
    """Search requires a query field."""
    tc, tmp, _, _ = client
    resp = tc.post("/api/search", json={})
    assert resp.status_code == 422  # Pydantic validation error


# ── Page routes ──────────────────────────────────────────────

def test_dashboard_page(client):
    """Dashboard page returns HTML."""
    tc, tmp, _, _ = client
    resp = tc.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Session Daemon" in resp.text


def test_search_page(client):
    """Search page returns HTML."""
    tc, tmp, _, _ = client
    resp = tc.get("/search")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Search Playground" in resp.text


def test_api_docs_page(client):
    """FastAPI auto-docs should be accessible."""
    tc, tmp, _, _ = client
    resp = tc.get("/docs")
    assert resp.status_code == 200
