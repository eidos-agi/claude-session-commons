"""Tests for chunkers.py — turn and subagent parsing."""

from pathlib import Path

from claude_session_commons.chunkers import (
    TurnChunk,
    SubagentChunk,
    chunk_turns,
    chunk_subagents,
    _extract_text,
    _extract_assistant_text,
    _extract_tools,
    _extract_files_touched,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_session.jsonl"


# ── Helper extraction tests ──────────────────────────────────

def test_extract_text_string():
    assert _extract_text("hello world") == "hello world"


def test_extract_text_array():
    content = [{"type": "text", "text": "hello"}, {"type": "image", "data": "..."}]
    assert _extract_text(content) == "hello"


def test_extract_text_empty():
    assert _extract_text("") == ""
    assert _extract_text([]) == ""
    assert _extract_text(None) == ""


def test_extract_assistant_text():
    content = [
        {"type": "text", "text": "First part"},
        {"type": "tool_use", "name": "Read", "input": {}},
        {"type": "text", "text": "Second part"},
    ]
    result = _extract_assistant_text(content)
    assert "First part" in result
    assert "Second part" in result


def test_extract_tools():
    content = [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "/foo/bar.py"}},
        {"type": "tool_use", "name": "Bash", "input": {"command": "git status"}},
        {"type": "text", "text": "some text"},
    ]
    names, descs = _extract_tools(content)
    assert names == ["Read", "Bash"]
    assert "Read(file_path=/foo/bar.py)" in descs
    assert "Bash(command=git status)" in descs


def test_extract_files_touched():
    content = [
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "/a.py", "old_string": "x"}},
        {"type": "tool_use", "name": "Write", "input": {"file_path": "/b.py", "content": "y"}},
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
    ]
    files = _extract_files_touched(content)
    assert "/a.py" in files
    assert "/b.py" in files
    assert len(files) == 2


# ── Turn chunker tests ───────────────────────────────────────

def test_chunk_turns_basic():
    """Parse fixture and verify we get the expected turn chunks."""
    chunks = chunk_turns(FIXTURE)
    assert isinstance(chunks, list)
    assert len(chunks) >= 4  # fixture has 6 user-assistant pairs

    # All chunks should be TurnChunk instances
    for c in chunks:
        assert isinstance(c, TurnChunk)
        assert c.content
        assert c.user_uuid
        assert c.assistant_uuid
        assert c.timestamp


def test_chunk_turns_content_format():
    """Verify content follows USER: / ASSISTANT: / TOOLS: format."""
    chunks = chunk_turns(FIXTURE)
    first = chunks[0]
    assert first.content.startswith("USER:")
    assert "ASSISTANT:" in first.content
    # First turn uses Glob and Read tools
    assert "TOOLS:" in first.content


def test_chunk_turns_metadata():
    """Verify metadata contains expected fields."""
    chunks = chunk_turns(FIXTURE)
    for c in chunks:
        m = c.metadata
        assert "user_uuid" in m
        assert "assistant_uuid" in m
        assert "model" in m
        assert "tools_used" in m
        assert "files_touched" in m
        assert "token_count" in m


def test_chunk_turns_truncation():
    """Content should never exceed MAX_CONTENT_CHARS."""
    chunks = chunk_turns(FIXTURE)
    for c in chunks:
        assert len(c.content) <= 2014  # 2000 + "[...truncated]"


def test_chunk_turns_tools_extracted():
    """Turns with tool calls should list them."""
    chunks = chunk_turns(FIXTURE)
    # The write migration turn should have Write tool
    write_turns = [c for c in chunks if "Write" in c.metadata.get("tools_used", [])]
    assert len(write_turns) >= 1


def test_chunk_turns_files_touched():
    """Turns with file edits should have files_touched populated."""
    chunks = chunk_turns(FIXTURE)
    file_turns = [c for c in chunks if c.metadata.get("files_touched")]
    assert len(file_turns) >= 1


# ── Subagent chunker tests ───────────────────────────────────

def test_chunk_subagents_basic():
    """Parse fixture for subagent chunks (without LLM summarization)."""
    chunks = chunk_subagents(FIXTURE, summarize=False)
    assert isinstance(chunks, list)
    # Fixture has 1 slug with 6 entries (resilient-roaming-frost) — above threshold
    # and 1 slug with 2 entries (gentle-amber-breeze) — below threshold
    assert len(chunks) == 1

    c = chunks[0]
    assert isinstance(c, SubagentChunk)
    assert c.slug == "resilient-roaming-frost"
    assert c.content
    assert c.timestamp


def test_chunk_subagents_metadata():
    """Verify subagent metadata fields."""
    chunks = chunk_subagents(FIXTURE, summarize=False)
    c = chunks[0]
    m = c.metadata
    assert m["slug"] == "resilient-roaming-frost"
    assert "initial_prompt_preview" in m
    assert "RBAC" in m["initial_prompt_preview"]
    assert m["progress_line_count"] == 6


def test_chunk_subagents_skips_small():
    """Subagents with < 5 entries should be skipped."""
    chunks = chunk_subagents(FIXTURE, summarize=False)
    slugs = [c.slug for c in chunks]
    assert "gentle-amber-breeze" not in slugs


def test_chunk_subagents_content_has_work():
    """Content should include the subagent's work text when not summarizing."""
    chunks = chunk_subagents(FIXTURE, summarize=False)
    c = chunks[0]
    assert "RBAC" in c.content or "role" in c.content.lower()
