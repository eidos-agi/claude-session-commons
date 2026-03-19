"""Tests for RAG agent tools and chat state."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from claude_session_commons.web.chat_state import ChatStateManager, ChatSession


# ── ChatStateManager tests ────────────────────────────────────

def test_create_session():
    mgr = ChatStateManager()
    chat_id = mgr.create_session()
    assert len(chat_id) == 8
    assert mgr.get_session(chat_id) is not None


def test_get_nonexistent_session():
    mgr = ChatStateManager()
    assert mgr.get_session("nope") is None


def test_add_message():
    mgr = ChatStateManager()
    chat_id = mgr.create_session()
    mgr.add_message(chat_id, "user", "hello")
    mgr.add_message(chat_id, "assistant", "hi there")
    session = mgr.get_session(chat_id)
    assert len(session.messages) == 2
    assert session.messages[0] == {"role": "user", "content": "hello"}
    assert session.messages[1] == {"role": "assistant", "content": "hi there"}


@pytest.mark.asyncio
async def test_event_queue():
    mgr = ChatStateManager()
    chat_id = mgr.create_session()
    await mgr.push_event(chat_id, {"type": "message", "html": "<p>hi</p>"})
    await mgr.push_event(chat_id, {"type": "done", "html": ""})

    events = []
    async for e in mgr.consume_events(chat_id):
        events.append(e)
    assert len(events) == 2
    assert events[0]["type"] == "message"
    assert events[1]["type"] == "done"


def test_ttl_cleanup():
    mgr = ChatStateManager()
    mgr.TTL_SECONDS = 0  # Expire immediately
    chat_id = mgr.create_session()
    import time
    time.sleep(0.01)
    mgr._cleanup_stale()
    assert mgr.get_session(chat_id) is None


def test_max_sessions_cleanup():
    mgr = ChatStateManager()
    mgr.MAX_SESSIONS = 3
    ids = [mgr.create_session() for _ in range(5)]
    # Only the most recent sessions should survive
    assert len(mgr._sessions) <= 3


# ── Tool rendering tests ────────────────────────────────────

def test_render_user_msg():
    from claude_session_commons.web.rag_agent import _render_user_msg
    html = _render_user_msg("hello <world>")
    assert "chat-msg user" in html
    assert "&lt;world&gt;" in html
    assert "<world>" not in html


def test_render_tool_call():
    from claude_session_commons.web.rag_agent import _render_tool_call
    html = _render_tool_call("session_search", {"query": "auth system"}, "tool-123")
    assert "tool-indicator" in html
    assert "tool-details" in html
    assert "session_search" in html
    assert "auth system" in html
    assert "tool-123" in html


def test_render_assistant_text():
    import base64
    from claude_session_commons.web.rag_agent import _render_assistant_text
    html = _render_assistant_text("Line 1\nLine 2")
    assert "chat-msg assistant" in html
    assert "markdown-pending" in html
    assert "data-raw-text" in html
    # Verify base64 encoding roundtrips
    b64 = html.split('data-raw-text="')[1].split('"')[0]
    assert base64.b64decode(b64).decode() == "Line 1\nLine 2"


def test_render_error():
    from claude_session_commons.web.rag_agent import _render_error
    html = _render_error("something broke")
    assert "chat-msg error" in html
    assert "something broke" in html


def test_format_history():
    from claude_session_commons.web.rag_agent import _format_history
    messages = [
        {"role": "user", "content": "question 1"},
        {"role": "assistant", "content": "answer 1"},
    ]
    result = _format_history(messages)
    assert "User: question 1" in result
    assert "Assistant: answer 1" in result


def test_format_history_empty():
    from claude_session_commons.web.rag_agent import _format_history
    assert _format_history([]) == ""


# ── Tool function tests (with mocked DB) ─────────────────────

try:
    import sqlite_vec
    from fastembed import TextEmbedding
    HAS_INSIGHTS = True
except ImportError:
    HAS_INSIGHTS = False


@pytest.mark.skipif(not HAS_INSIGHTS, reason="insights deps not installed")
@pytest.mark.asyncio
async def test_db_overview_tool():
    from claude_session_commons.web.rag_agent import db_overview
    from claude_session_commons.insights import get_db, init_db

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        conn = get_db(db_path)
        init_db(conn)
        conn.close()

        with patch("claude_session_commons.web.rag_agent.DB_PATH", db_path):
            result = await db_overview.handler({})

        text = result["content"][0]["text"]
        assert "Sessions indexed" in text
        assert "Total chunks" in text


@pytest.mark.skipif(not HAS_INSIGHTS, reason="insights deps not installed")
@pytest.mark.asyncio
async def test_session_search_no_results():
    from claude_session_commons.web.rag_agent import session_search
    from claude_session_commons.insights import get_db, init_db

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        conn = get_db(db_path)
        init_db(conn)
        conn.close()

        with patch("claude_session_commons.web.rag_agent.DB_PATH", db_path):
            result = await session_search.handler({"query": "nonexistent"})

        text = result["content"][0]["text"]
        assert "No results" in text


@pytest.mark.skipif(not HAS_INSIGHTS, reason="insights deps not installed")
@pytest.mark.asyncio
async def test_entity_search_no_results():
    from claude_session_commons.web.rag_agent import entity_search
    from claude_session_commons.insights import get_db, init_db

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        conn = get_db(db_path)
        init_db(conn)
        conn.close()

        with patch("claude_session_commons.web.rag_agent.DB_PATH", db_path):
            result = await entity_search.handler({"entity_type": "file_path", "value": "nothing"})

        text = result["content"][0]["text"]
        assert "No results" in text
