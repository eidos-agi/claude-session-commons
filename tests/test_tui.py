"""Automated Textual pilot tests for the shared SessionPickerPanel."""

import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, ListView, Static

from claude_session_commons.tui import SessionPickerPanel, SessionOps
from claude_session_commons.tui.widgets import SearchInput, DateHeader, SessionItem, PreviewMode
from claude_session_commons.cache import SessionCache


# ── Fixtures ─────────────────────────────────────────────


def _make_session(session_id: str, project_dir: str, age_hours: float = 1, size: int = 5000):
    """Create a fake session dict matching find_all_sessions() output."""
    mtime = time.time() - (age_hours * 3600)
    return {
        "session_id": session_id,
        "project_dir": project_dir,
        "file": Path(f"/tmp/fake-sessions/{session_id}.jsonl"),
        "mtime": mtime,
        "size": size,
        "last_entry_type": "assistant",
    }


def _make_summary(title: str = "Test Session"):
    return {
        "title": title,
        "goal": "Test goal",
        "what_was_done": "Test work done",
        "state": "Test state",
        "files": ["/tmp/test.py"],
    }


def _mock_ops(tmp_path, sessions=None):
    """Build SessionOps with mock callables.

    If *sessions* is provided, pre-populate the cache with stats and search_text
    so that background classify/index workers have nothing to do.
    """
    cache = SessionCache(tmp_path / "cache")
    ops = SessionOps(
        cache=cache,
        parse_session=MagicMock(return_value=({"first_messages": [], "last_messages": [],
                                                "last_assistant": [], "first_assistant": [],
                                                "recent_tools": [], "all_tools": [],
                                                "total_user_messages": 5, "total_lines": 100},
                                               "search text")),
        get_git_context=MagicMock(return_value={"is_git_repo": False}),
        summarize_quick=MagicMock(return_value=_make_summary("Quick Summary")),
        summarize_deep=MagicMock(return_value={
            "title": "Deep Title", "objective": "Deep obj", "progress": "Done",
            "state": "Left off", "next_steps": "Do more", "files": [], "decisions_made": [],
        }),
        analyze_patterns=MagicMock(return_value={
            "prompt_patterns": {"effective": [], "ineffective": [], "tips": ["tip1"]},
            "workflow_patterns": {"common_sequences": [], "iteration_style": "iterative"},
            "anti_patterns": [], "key_lesson": "test lesson",
        }),
    )
    # Pre-cache stats and search text so background workers are no-ops
    if sessions:
        for s in sessions:
            ck = cache.cache_key(s["file"])
            cache.set(s["session_id"], ck, "stats", {
                "classification": "interactive",
                "user_messages": 5, "assistant_messages": 10,
                "tool_uses": 3, "tool_results": 3,
                "duration_fmt": "1h 0m",
            })
            cache.set(s["session_id"], ck, "search_text",
                      f"{s['project_dir']} {s['session_id']}".lower())
    return ops


class PickerTestApp(App):
    """Minimal App wrapper for testing SessionPickerPanel."""

    CSS = "Screen { layout: vertical; }"

    def __init__(self, sessions, summaries, ops):
        super().__init__()
        self._sessions = sessions
        self._summaries = summaries
        self._ops = ops
        self.result_data = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield SessionPickerPanel(
            self._sessions, self._summaries, self._ops,
            title="test-app", id="picker",
        )
        yield Footer()

    def on_session_picker_panel_session_selected(self, message):
        self.result_data = (message.action, message.idx, message.cmd, message.cmds)
        self.exit()

    def on_key(self, event):
        # Panel handles session-specific keys via its own on_key (event bubbling).
        # We only handle keys the panel doesn't consume.
        if event.key == "escape" or event.character == "q":
            self.exit()
            event.prevent_default()
            event.stop()


# ── Tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_panel_renders_sessions(tmp_path):
    """Panel should render session items with date groups."""
    sessions = [
        _make_session("aaa", "/tmp/project-a", age_hours=1),
        _make_session("bbb", "/tmp/project-b", age_hours=1),
    ]
    summaries = [_make_summary("Session A"), _make_summary("Session B")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # Panel should be mounted
        panel = app.query_one("#picker", SessionPickerPanel)
        assert panel is not None

        # Should have a ListView with items
        lv = app.query_one("#sp-session-list", ListView)
        assert len(lv.children) >= 2  # At least date header + 2 sessions

        # Should have at least one DateHeader
        headers = app.query(DateHeader)
        assert len(headers) >= 1

        # Should have SessionItems
        items = app.query(SessionItem)
        assert len(items) == 2


@pytest.mark.asyncio
async def test_panel_renders_empty(tmp_path):
    """Panel with no sessions shows 'No matching' message."""
    ops = _mock_ops(tmp_path)
    app = PickerTestApp([], [], ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        preview = app.query_one("#sp-preview", Static)
        assert "No matching" in preview.render().plain


@pytest.mark.asyncio
async def test_search_filters_sessions(tmp_path):
    """Typing in search should filter the session list."""
    sessions = [
        _make_session("aaa", "/tmp/project-alpha", age_hours=1),
        _make_session("bbb", "/tmp/project-beta", age_hours=1),
    ]
    summaries = [_make_summary("Alpha Work"), _make_summary("Beta Work")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # Initially 2 sessions
        items = app.query(SessionItem)
        assert len(items) == 2

        # Focus search and type filter
        await pilot.press("slash")
        await pilot.pause()
        await pilot.press("a", "l", "p", "h", "a")
        await pilot.pause()

        # Should filter to just alpha
        items = app.query(SessionItem)
        assert len(items) == 1


@pytest.mark.asyncio
async def test_escape_from_search(tmp_path):
    """Pressing Escape in search should clear it and return to list."""
    sessions = [_make_session("aaa", "/tmp/proj", age_hours=1)]
    summaries = [_make_summary("Test")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # Open search
        await pilot.press("slash")
        search = app.query_one("#sp-search", SearchInput)
        assert search == app.focused

        # Type something
        await pilot.press("x", "y", "z")
        assert search.value == "xyz"

        # Escape clears search
        await pilot.press("escape")
        assert search.value == ""


@pytest.mark.asyncio
async def test_preview_updates_on_highlight(tmp_path):
    """Highlighting a session should update the preview pane."""
    sessions = [
        _make_session("aaa", "/tmp/project-a", age_hours=1),
        _make_session("bbb", "/tmp/project-b", age_hours=2),
    ]
    summaries = [_make_summary("Session Alpha"), _make_summary("Session Beta")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        preview = app.query_one("#sp-preview", Static)
        content = preview.render().plain
        # First session should be in preview
        assert "Session Alpha" in content


@pytest.mark.asyncio
async def test_skip_permissions_toggle(tmp_path):
    """Pressing 'd' should toggle skip-permissions in preview."""
    sessions = [_make_session("aaa", "/tmp/proj", age_hours=1)]
    summaries = [_make_summary("Test")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one("#picker", SessionPickerPanel)
        assert panel._skip_permissions is True

        await pilot.press("d")
        assert panel._skip_permissions is False

        await pilot.press("d")
        assert panel._skip_permissions is True


@pytest.mark.asyncio
async def test_multi_select(tmp_path):
    """Space should toggle selection, subtitle should update."""
    sessions = [
        _make_session("aaa", "/tmp/proj-a", age_hours=1),
        _make_session("bbb", "/tmp/proj-b", age_hours=1),
    ]
    summaries = [_make_summary("A"), _make_summary("B")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one("#picker", SessionPickerPanel)
        assert len(panel._selected) == 0

        # Select first
        await pilot.press("space")
        await pilot.pause()
        assert len(panel._selected) == 1

        # Deselect
        await pilot.press("space")
        await pilot.pause()
        assert len(panel._selected) == 0


@pytest.mark.asyncio
async def test_toggle_bots(tmp_path):
    """Pressing 'b' should toggle show_bots."""
    sessions = [_make_session("aaa", "/tmp/proj", age_hours=1)]
    summaries = [_make_summary("Test")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one("#picker", SessionPickerPanel)
        assert panel._show_bots is False

        await pilot.press("b")
        assert panel._show_bots is True

        await pilot.press("b")
        assert panel._show_bots is False


@pytest.mark.asyncio
async def test_preview_pane_focus(tmp_path):
    """Right arrow should focus preview, left should return."""
    sessions = [_make_session("aaa", "/tmp/proj", age_hours=1)]
    summaries = [_make_summary("Test")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one("#picker", SessionPickerPanel)
        assert panel._in_preview is False

        await pilot.press("right")
        assert panel._in_preview is True

        scroll = app.query_one("#sp-preview-scroll")
        assert scroll.has_class("focused")

        await pilot.press("left")
        assert panel._in_preview is False
        assert not scroll.has_class("focused")


@pytest.mark.asyncio
async def test_quit_on_q(tmp_path):
    """Pressing 'q' should exit the app."""
    sessions = [_make_session("aaa", "/tmp/proj", age_hours=1)]
    summaries = [_make_summary("Test")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("q")
    assert app.result_data is None


@pytest.mark.asyncio
async def test_enter_selects_session(tmp_path):
    """Pressing Enter on a session should emit SessionSelected."""
    sessions = [_make_session("aaa", "/tmp/proj", age_hours=1)]
    summaries = [_make_summary("Test")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
    assert app.result_data is not None
    action, idx, cmd, cmds = app.result_data
    assert action == "select"
    assert "claude --resume aaa" in cmd


@pytest.mark.asyncio
async def test_r_resumes_session(tmp_path):
    """Pressing 'r' should emit resume action."""
    sessions = [_make_session("aaa", "/tmp/proj", age_hours=1)]
    summaries = [_make_summary("Test")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("r")
    assert app.result_data is not None
    action, idx, cmd, cmds = app.result_data
    assert action == "resume"
    assert "claude --resume aaa" in cmd
    assert "--dangerously-skip-permissions" in cmd


@pytest.mark.asyncio
async def test_patterns_toggle(tmp_path):
    """Pressing 'p' should toggle patterns preview mode."""
    sessions = [_make_session("aaa", "/tmp/proj", age_hours=1)]
    summaries = [_make_summary("Test")]
    ops = _mock_ops(tmp_path, sessions)

    app = PickerTestApp(sessions, summaries, ops)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one("#picker", SessionPickerPanel)
        assert panel._preview_mode == PreviewMode.SUMMARY

        await pilot.press("p")
        assert panel._preview_mode == PreviewMode.PATTERNS

        # Wait a moment for the background task
        await pilot.pause(delay=0.1)

        await pilot.press("p")
        assert panel._preview_mode == PreviewMode.SUMMARY
