"""Reusable session picker panel — shared between claude-resume and claude-boss.

This is a Textual Widget that provides:
- Session list with date grouping and classification filtering
- Background AI summarization, deep dive, patterns analysis
- Search with background indexing
- Preview pane with session stats, summaries, deep analysis
- Multi-select, export, toggle bots
- All key bindings (D, p, x, Space, b, r, d, /)
"""

import json
import os
import queue
import threading
import time as _time

from pathlib import Path
from rich.markup import MarkupError
from textual import events, on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, ListView, Static

from ..classify import get_label
from ..display import get_date_group, lifecycle_badge, relative_time
from ..export import export_context_md
from ..paths import shorten_path

from .ops import SessionOps
from .widgets import DateHeader, PreviewMode, SearchInput, SessionItem, TaskDone


_LLM_TASKS = frozenset({"summarize", "deep", "patterns"})


def esc(text: str) -> str:
    """Escape ALL [ chars so Rich never interprets user content as markup."""
    return str(text).replace("[", "\\[")


class SessionPickerPanel(Widget):
    """Embeddable session picker with full functionality.

    Usage:
        panel = SessionPickerPanel(sessions, summaries, ops)
        # Mount it in your app's compose()
        # Listen for SessionPickerPanel.SessionSelected messages

    Messages emitted:
        SessionSelected — user chose a session (single or multi)
    """

    DEFAULT_CSS = """
    SessionPickerPanel { height: 1fr; width: 1fr; layout: vertical; }
    SessionPickerPanel #sp-search { dock: top; margin: 0 1; height: 3; }
    SessionPickerPanel #sp-main { height: 1fr; }
    SessionPickerPanel #sp-session-list { width: 45%; border-right: heavy $primary; }
    SessionPickerPanel #sp-preview-scroll { width: 55%; padding: 1 2; overflow-y: auto; }
    SessionPickerPanel #sp-preview-scroll.focused { border-left: heavy $accent; padding: 1 1; }
    SessionPickerPanel #sp-preview { width: 100%; }
    SessionPickerPanel DateHeader { height: auto; padding: 1 0 0 1; }
    """

    class SessionSelected(Message):
        """Emitted when user selects session(s)."""
        def __init__(self, action: str, idx: int, cmd: str | None = None,
                     cmds: list[str] | None = None) -> None:
            super().__init__()
            self.action = action  # "select", "resume", "multi_resume"
            self.idx = idx
            self.cmd = cmd
            self.cmds = cmds

    def __init__(self, sessions: list, summaries: list, ops: SessionOps,
                 title: str = "claude-resume",
                 placeholder: str = "/ search  r resume  Space select  x export  d perms  D dive  p patterns  b bots  C chat  Esc quit",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.sessions = sessions
        self.summaries = summaries
        self._ops = ops
        self._picker_title = title
        self._placeholder = placeholder
        self.filtered_items: list[tuple[int, dict, dict | None]] = []
        self._saved_session_idx = 0
        self._skip_permissions = True
        self._show_bots = False
        self._selected: set[int] = set()
        self._in_preview = False
        self._preview_mode = PreviewMode.SUMMARY
        self._lv_map: dict[int, int] = {}
        self._last_lv_index = 0
        self._search_index: list[str] = []
        self._pending: dict[str, set[int]] = {
            "summarize": set(), "deep": set(), "patterns": set(), "index": set(), "scan": set(),
        }
        # Serial queue for LLM tasks — one claude -p call at a time
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._queue_worker: threading.Thread | None = None
        self._active_task: tuple[str, int] | None = None
        self._queue_counter = 0

    def compose(self) -> ComposeResult:
        yield SearchInput(
            placeholder=self._placeholder,
            id="sp-search",
        )
        with Horizontal(id="sp-main"):
            yield ListView(id="sp-session-list")
            with VerticalScroll(id="sp-preview-scroll"):
                yield Static("", id="sp-preview", markup=True)

    # ── Lifecycle ──────────────────────────────────────────

    def on_mount(self) -> None:
        self._init_search_index()
        self._classify_uncached()
        self._enqueue_unsummarized()
        self._populate_list()
        self.query_one("#sp-session-list", ListView).focus()

    def _enqueue_unsummarized(self) -> None:
        """Queue all unsummarized sessions, most recent first."""
        for i, s in enumerate(self.sessions):
            if self.summaries[i] is not None:
                continue
            ck = self._ops.cache.cache_key(s["file"])
            if self._ops.cache.get(s["session_id"], ck, "summary") is not None:
                continue
            self._start_task("summarize", i)

    def _classify_uncached(self) -> None:
        unscanned = []
        cache = self._ops.cache
        for i, s in enumerate(self.sessions):
            ck = cache.cache_key(s["file"])
            if cache.get(s["session_id"], ck, "stats") is None:
                unscanned.append(i)
        if unscanned:
            # Most recent first
            unscanned.sort(key=lambda i: self.sessions[i]["mtime"], reverse=True)
            self._classify_batch_bg(unscanned)

    @work(thread=True)
    def _classify_batch_bg(self, indices: list[int]) -> None:
        for i in indices:
            s = self.sessions[i]
            try:
                get_label(s["file"], self._ops.cache)
            except Exception:
                pass
        self.post_message(TaskDone("scan", 0, None))

    def _init_search_index(self) -> None:
        self._search_index = []
        uncached = []
        for i, s in enumerate(self.sessions):
            ck = self._ops.cache.cache_key(s["file"])
            cached = self._ops.cache.get(s["session_id"], ck, "search_text")
            self._search_index.append(cached or "")
            if not cached:
                uncached.append(i)
        if uncached:
            # Most recent first
            uncached.sort(key=lambda i: self.sessions[i]["mtime"], reverse=True)
            self._index_batch_bg(uncached)

    # ── Background task engine ─────────────────────────────

    def _start_task(self, kind: str, idx: int) -> None:
        if idx in self._pending[kind]:
            return
        self._pending[kind].add(idx)
        if kind in _LLM_TASKS:
            # LLM tasks go through the serial queue — one at a time, most recent first
            mtime = self.sessions[idx].get("mtime", 0)
            self._queue_counter += 1
            self._task_queue.put((-mtime, self._queue_counter, kind, idx))
            self._ensure_worker()
        else:
            self._run_task_bg(kind, idx)

    def _ensure_worker(self) -> None:
        """Start the queue worker thread if not already running."""
        if self._queue_worker is None or not self._queue_worker.is_alive():
            self._queue_worker = threading.Thread(
                target=self._process_queue, daemon=True,
            )
            self._queue_worker.start()

    def _process_queue(self) -> None:
        """Worker loop: pull tasks one at a time until queue is empty."""
        while True:
            try:
                _, _, kind, idx = self._task_queue.get(timeout=2.0)
            except queue.Empty:
                self._active_task = None
                return
            self._active_task = (kind, idx)
            self._execute_task(kind, idx)
            self._active_task = None
            self._task_queue.task_done()

    # ── Daemon delegation ─────────────────────────────────

    _DAEMON_PID_FILE = Path.home() / ".claude" / "session-daemon.pid"
    _DAEMON_TASK_DIR = Path.home() / ".claude" / "daemon-tasks"
    _CACHE_FIELD = {"summarize": "summary", "deep": "deep_summary", "patterns": "patterns"}

    def _daemon_alive(self) -> bool:
        """Check if the background daemon is running."""
        try:
            if not self._DAEMON_PID_FILE.exists():
                return False
            pid = int(self._DAEMON_PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            return False

    def _write_task_file(self, kind: str, session: dict, quick_summary=None) -> None:
        """Write a task request for the daemon to pick up."""
        self._DAEMON_TASK_DIR.mkdir(parents=True, exist_ok=True)
        priority = int(_time.time() * 1000)
        filename = f"{priority}-{kind}-{session['session_id'][:8]}.json"
        task = {
            "kind": kind,
            "session_id": session["session_id"],
            "file": str(session["file"]),
            "project_dir": session["project_dir"],
            "quick_summary": quick_summary,
        }
        (self._DAEMON_TASK_DIR / filename).write_text(json.dumps(task))

    def _poll_cache_for_result(self, kind: str, session_id: str,
                               cache_key: str, timeout: int = 120):
        """Poll cache until daemon writes the result. Returns value or None."""
        field = self._CACHE_FIELD[kind]
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            result = self._ops.cache.get(session_id, cache_key, field)
            if result is not None:
                return result
            _time.sleep(1)
        return None

    def _execute_task(self, kind: str, idx: int) -> None:
        """Run a single LLM task — delegates to daemon if alive, else local."""
        s = self.sessions[idx]
        ops = self._ops
        ck = ops.cache.cache_key(s["file"])

        # Try daemon delegation first
        if self._daemon_alive():
            quick = self.summaries[idx] if kind in ("deep", "patterns") else None
            self._write_task_file(kind, s, quick_summary=quick)
            result = self._poll_cache_for_result(kind, s["session_id"], ck)
            if result is not None:
                if kind == "summarize":
                    search = ops.cache.get(s["session_id"], ck, "search_text") or ""
                    self.post_message(TaskDone(kind, idx, {"summary": result, "search_text": search}))
                else:
                    self.post_message(TaskDone(kind, idx, result))
                return

        # Local fallback (daemon dead or timed out)
        try:
            if kind == "summarize":
                context, search_text = ops.parse_session(s["file"])
                git = ops.get_git_context(s["project_dir"])
                summary = ops.summarize_quick(context, s["project_dir"], git)
                ops.cache.set(s["session_id"], ck, "summary", summary)
                full = (search_text + f" {s['project_dir']} {s['session_id']}").lower()
                ops.cache.set(s["session_id"], ck, "search_text", full)
                if "stats" in context:
                    ops.cache.set(s["session_id"], ck, "stats", context["stats"])
                self.post_message(TaskDone(kind, idx, {"summary": summary, "search_text": full}))

            elif kind == "deep":
                context, _ = ops.parse_session(s["file"], deep=True)
                git = ops.get_git_context(s["project_dir"])
                quick = self.summaries[idx] or {"title": "Unknown", "state": "", "files": []}
                deep = ops.summarize_deep(context, s["project_dir"], quick, git)
                ops.cache.set(s["session_id"], ck, "deep_summary", deep)
                self.post_message(TaskDone(kind, idx, deep))

            elif kind == "patterns":
                context, _ = ops.parse_session(s["file"], deep=True)
                quick = self.summaries[idx] or {"title": "Unknown", "state": "", "files": []}
                patterns = ops.analyze_patterns(context, s["project_dir"], quick)
                ops.cache.set(s["session_id"], ck, "patterns", patterns)
                self.post_message(TaskDone(kind, idx, patterns))

        except Exception as e:
            self.post_message(TaskDone(kind, idx, error=str(e)))

    @work(thread=True)
    def _run_task_bg(self, kind: str, idx: int) -> None:
        """Background worker for non-LLM tasks (scan, index)."""
        s = self.sessions[idx]
        ops = self._ops
        ck = ops.cache.cache_key(s["file"])
        try:
            if kind == "scan":
                get_label(s["file"], ops.cache)
                scan = ops.cache.get(s["session_id"], ck, "stats") or {}
                self.post_message(TaskDone(kind, idx, scan))

            elif kind == "index":
                _, search_text = ops.parse_session(s["file"])
                full = (search_text + f" {s['project_dir']} {s['session_id']}").lower()
                ops.cache.set(s["session_id"], ck, "search_text", full)
                self.post_message(TaskDone(kind, idx, full))

        except Exception as e:
            self.post_message(TaskDone(kind, idx, error=str(e)))

    @work(thread=True)
    def _index_batch_bg(self, indices: list[int]) -> None:
        for i in indices:
            s = self.sessions[i]
            try:
                _, search_text = self._ops.parse_session(s["file"])
                full = (search_text + f" {s['project_dir']} {s['session_id']}").lower()
                ck = self._ops.cache.cache_key(s["file"])
                self._ops.cache.set(s["session_id"], ck, "search_text", full)
                self.post_message(TaskDone("index", i, full))
            except Exception as e:
                self.post_message(TaskDone("index", i, error=str(e)))

    def on_task_done(self, message: TaskDone) -> None:
        self._pending[message.kind].discard(message.idx)
        if not self.is_attached:
            return

        if message.kind == "summarize":
            if message.error:
                self.summaries[message.idx] = {
                    "title": "Summary failed", "goal": "", "what_was_done": "",
                    "state": f"Error: {message.error}", "files": [],
                }
            else:
                self.summaries[message.idx] = message.result["summary"]
                if message.idx < len(self._search_index):
                    self._search_index[message.idx] = message.result["search_text"]
            self._refresh_list()

        elif message.kind == "deep":
            fi = self._current_filtered_index()
            if fi is not None and self.filtered_items[fi][0] == message.idx:
                if message.error:
                    self._show_preview_error(f"Deep dive failed: {message.error}")
                elif self._preview_mode == PreviewMode.SUMMARY:
                    self._update_preview(fi)
            self._refresh_list()

        elif message.kind == "patterns":
            fi = self._current_filtered_index()
            if (self._preview_mode == PreviewMode.PATTERNS
                    and fi is not None
                    and self.filtered_items[fi][0] == message.idx):
                if message.error:
                    self._show_preview_error(f"Patterns analysis failed: {message.error}")
                else:
                    self._display_patterns(message.idx, message.result)

        elif message.kind == "index":
            if not message.error and message.idx < len(self._search_index):
                self._search_index[message.idx] = message.result

        elif message.kind == "scan":
            self._refresh_list()

    # ── List management ────────────────────────────────────

    def _refresh_list(self) -> None:
        if not self.is_attached:
            return
        search = self.query_one("#sp-search", SearchInput)
        self._populate_list(search.value)

    def _populate_list(self, filter_text: str = "") -> None:
        query = filter_text.lower()
        cache = self._ops.cache
        self.filtered_items = []
        hidden_count = 0
        for i, (s, sm) in enumerate(zip(self.sessions, self.summaries)):
            ck = cache.cache_key(s["file"])
            deep = cache.get(s["session_id"], ck, "deep_summary")
            best = deep or sm

            if not self._show_bots:
                stats = cache.get(s["session_id"], ck, "stats")
                if stats and stats.get("classification") == "automated":
                    hidden_count += 1
                    continue

            if not query or (i < len(self._search_index) and query in self._search_index[i]):
                self.filtered_items.append((i, s, best))

        lv = self.query_one("#sp-session-list", ListView)

        # Remove existing children synchronously to avoid async clear/mount races
        for child in list(lv.children):
            child.remove()

        self._lv_map = {}
        current_group = None
        lv_idx = 0
        items_to_add = []

        for fi, (idx, session, summary) in enumerate(self.filtered_items):
            group = get_date_group(session["mtime"])
            if group != current_group:
                current_group = group
                items_to_add.append(DateHeader(group))
                lv_idx += 1
            ck = cache.cache_key(session["file"])
            has_deep = cache.get(session["session_id"], ck, "deep_summary") is not None
            is_summarizing = (self._active_task is not None
                              and self._active_task[0] == "summarize"
                              and self._active_task[1] == idx)
            bookmark = cache.get(session["session_id"], ck, "bookmark")
            self._lv_map[lv_idx] = fi
            items_to_add.append(SessionItem(idx, session, summary, has_deep=has_deep,
                                            selected=idx in self._selected,
                                            is_summarizing=is_summarizing,
                                            bookmark=bookmark))
            lv_idx += 1

        if hidden_count and not self._show_bots:
            items_to_add.append(DateHeader(f"{hidden_count} automated sessions hidden (b to show)"))
            lv_idx += 1

        if items_to_add:
            lv.mount(*items_to_add)

        if self.filtered_items:
            target_lv = None
            if not query:
                for lv_i, fi in self._lv_map.items():
                    if self.filtered_items[fi][0] == self._saved_session_idx:
                        target_lv = lv_i
                        break
            if target_lv is None:
                target_lv = min(self._lv_map.keys()) if self._lv_map else 0
            lv.index = target_lv
            self._last_lv_index = target_lv
        else:
            self.query_one("#sp-preview", Static).update("[dim]No matching sessions[/]")

    def _current_filtered_index(self) -> int | None:
        lv = self.query_one("#sp-session-list", ListView)
        if lv.index is not None and lv.index in self._lv_map:
            return self._lv_map[lv.index]
        return None

    # ── Preview rendering ──────────────────────────────────

    def _update_preview(self, filtered_idx: int) -> None:
        if filtered_idx < 0 or filtered_idx >= len(self.filtered_items):
            self.query_one("#sp-preview", Static).update("")
            return

        orig_idx, session, summary = self.filtered_items[filtered_idx]

        if summary is None:
            short_path = shorten_path(session["project_dir"])
            age = relative_time(session["mtime"])
            is_active = (self._active_task is not None
                         and self._active_task[0] == "summarize"
                         and self._active_task[1] == orig_idx)
            status = "Summarizing..." if is_active else "Queued..."
            self.query_one("#sp-preview", Static).update(
                f"[bold yellow]{status}[/]\n\n"
                f"[bold]Directory:[/]   [cyan]{short_path}[/]\n"
                f"[bold]Last active:[/] {age}\n"
            )
            return

        cache = self._ops.cache
        ck = cache.cache_key(session["file"])
        deep = cache.get(session["session_id"], ck, "deep_summary")
        display = deep or summary

        short_path = shorten_path(session["project_dir"])
        age = relative_time(session["mtime"])
        size_mb = session["size"] / (1024 * 1024)

        title = esc(display.get("title", summary.get("title", "Unknown")))
        goal = esc(display.get("goal", summary.get("goal", "")))
        what_was_done = esc(display.get("what_was_done", summary.get("what_was_done", "")))
        state = esc(display.get("state", summary.get("state", "No context")))
        files = display.get("files", summary.get("files", []))

        diving = orig_idx in self._pending["deep"]
        analyzing = orig_idx in self._pending["patterns"]
        perms_status = "[green]ON[/]" if self._skip_permissions else "[red]OFF[/]"

        status_parts = []
        if diving:
            status_parts.append("[bold yellow]⟳ Deep analyzing...[/]")
        if analyzing:
            status_parts.append("[bold yellow]⟳ Analyzing patterns...[/]")
        status = "  " + " ".join(status_parts) if status_parts else ""

        stats = cache.get(session["session_id"], ck, "stats")
        bookmark = cache.get(session["session_id"], ck, "bookmark")
        lc = lifecycle_badge(bookmark) if bookmark and isinstance(bookmark, dict) else ""
        lc_line = f"  {lc}" if lc else ""

        text = f"""[bold underline]{title}[/]{lc_line}{status}

[bold]Directory:[/]   [cyan]{short_path}[/]
[bold]Last active:[/] {age}
[bold]Size:[/]        {size_mb:.1f} MB
[bold]Skip perms:[/]  {perms_status}  [dim](d to toggle)[/]
"""
        if stats:
            dur = stats.get("duration_fmt", "?")
            cls = stats.get("classification", "?")
            cls_color = "green" if cls == "interactive" else "dim"
            text += f"\n[bold]Session stats:[/]\n"
            text += f"  Duration:     {dur}\n"
            text += f"  User msgs:    {stats.get('user_messages', '?')}\n"
            text += f"  Asst msgs:    {stats.get('assistant_messages', '?')}\n"
            text += f"  Tool uses:    {stats.get('tool_uses', '?')}\n"
            text += f"  Tool results: {stats.get('tool_results', '?')}\n"
            if stats.get("system_entries"):
                text += f"  System:       {stats['system_entries']}\n"
            if stats.get("progress_entries"):
                text += f"  Progress:     {stats['progress_entries']}\n"
            text += f"  Type:         [{cls_color}]{cls}[/]\n"
        if goal:
            text += f"\n[bold]Goal:[/]\n{goal}\n"
        if what_was_done:
            text += f"\n[bold]What was done:[/]\n{what_was_done}\n"
        text += f"\n[bold]Where you left off:[/]\n{state}\n"

        # Bookmark: next actions and blockers injected right after state
        if bookmark and isinstance(bookmark, dict):
            bk_blockers = bookmark.get("blockers", [])
            if bk_blockers:
                text += "\n[bold red]Blockers:[/]\n"
                for b in bk_blockers:
                    desc = b.get("description", str(b)) if isinstance(b, dict) else str(b)
                    text += f"  [red]•[/] {esc(desc)}\n"
            bk_next = bookmark.get("next_actions", [])
            if bk_next:
                text += "\n[bold]Next actions:[/]\n"
                for i, a in enumerate(bk_next, 1):
                    text += f"  {i}. {esc(a)}\n"
            bk_conf = bookmark.get("confidence", {})
            if bk_conf and bk_conf.get("level"):
                level = bk_conf["level"]
                color = {"high": "green", "medium": "yellow", "low": "red"}.get(level, "dim")
                text += f"\n[bold]Confidence:[/] [{color}]{level}[/]"
                for r in bk_conf.get("risk_areas", []):
                    text += f"  [dim]({esc(r)})[/]"
                text += "\n"

        if files:
            text += "\n[bold]Key files:[/]\n"
            for f in files[:5]:
                text += f"  [dim]•[/] {esc(str(f))}\n"

        if deep:
            objective = esc(deep.get("objective", ""))
            if objective:
                text += f"\n[bold]Objective:[/]\n{objective}\n"
            progress = esc(deep.get("progress", ""))
            if progress:
                text += f"\n[bold]Progress:[/]\n{progress}\n"
            next_steps = esc(deep.get("next_steps", ""))
            if next_steps:
                text += f"\n[bold]Next steps:[/]\n{next_steps}\n"
            decisions = deep.get("decisions_made", [])
            if decisions:
                text += "\n[bold]Decisions:[/]\n"
                for d in decisions:
                    text += f"  [dim]•[/] {esc(d)}\n"

        self._safe_preview_update(text)

    def _safe_preview_update(self, text: str) -> None:
        """Update preview static with markup error recovery."""
        preview = self.query_one("#sp-preview", Static)
        try:
            preview.update(text)
        except MarkupError:
            preview.update(esc(text))
        self.query_one("#sp-preview-scroll", VerticalScroll).scroll_home(animate=False)

    def _display_patterns(self, orig_idx: int, patterns: dict) -> None:
        session = self.sessions[orig_idx]
        short_path = shorten_path(session["project_dir"])

        text = f"[bold yellow]━━━ Patterns: {short_path} ━━━[/]\n\n"

        pp = patterns.get("prompt_patterns", {})
        effective = pp.get("effective", [])
        ineffective = pp.get("ineffective", [])
        tips = pp.get("tips", [])

        if effective:
            text += "[bold green]Effective prompts:[/]\n"
            for e in effective:
                text += f'  [green]✓[/] "{esc(e.get("example", ""))}"\n'
                text += f'    {esc(e.get("why", ""))}\n'
            text += "\n"

        if ineffective:
            text += "[bold red]Ineffective prompts:[/]\n"
            for e in ineffective:
                text += f'  [red]✗[/] "{esc(e.get("example", ""))}"\n'
                text += f'    {esc(e.get("issue", ""))}\n'
            text += "\n"

        if tips:
            text += "[bold cyan]Tips:[/]\n"
            for t in tips:
                text += f"  • {esc(t)}\n"
            text += "\n"

        wp = patterns.get("workflow_patterns", {})
        sequences = wp.get("common_sequences", [])
        style = wp.get("iteration_style", "")
        if style:
            text += f"[bold]Iteration style:[/] {esc(style)}\n"
        if sequences:
            text += "[bold]Common tool sequences:[/]\n"
            for seq in sequences:
                tools = " → ".join(esc(t) for t in seq.get("tools", []))
                eff = esc(seq.get("efficiency", ""))
                ctx = esc(seq.get("context", ""))
                text += f"  \\[{eff}] {tools}\n"
                if ctx:
                    text += f"         {ctx}\n"
            text += "\n"

        anti = patterns.get("anti_patterns", [])
        if anti:
            text += "[bold red]Anti-patterns:[/]\n"
            for a in anti:
                text += f"  ⚠ {esc(a.get('pattern', ''))}\n"
                text += f"    Cost: {esc(a.get('cost', ''))}\n"
                text += f"    Fix:  {esc(a.get('fix', ''))}\n"
            text += "\n"

        lesson = patterns.get("key_lesson", "")
        if lesson:
            text += f"[bold magenta]Key lesson:[/] {esc(lesson)}\n"

        text += "\n[dim]Press p to return to summary[/]"

        self._safe_preview_update(text)

    def _show_preview_error(self, msg: str) -> None:
        self._safe_preview_update(
            f"[bold red]{esc(msg)}[/]\n\n[dim]Will retry on next run[/]"
        )

    # ── Search events ──────────────────────────────────────

    @on(Input.Changed, "#sp-search")
    def on_search_changed(self, event: Input.Changed) -> None:
        self._populate_list(event.value)

    @on(Input.Submitted, "#sp-search")
    def on_search_submit(self, event: Input.Submitted) -> None:
        self.query_one("#sp-session-list", ListView).focus()

    @on(SearchInput.Escaped)
    def on_search_escaped(self, event: SearchInput.Escaped) -> None:
        self._populate_list("")
        self.query_one("#sp-session-list", ListView).focus()

    # ── List events ────────────────────────────────────────

    @on(ListView.Highlighted)
    def on_highlight(self, event: ListView.Highlighted) -> None:
        lv = self.query_one("#sp-session-list", ListView)
        if lv.index is None:
            return
        if lv.index not in self._lv_map:
            if lv.index > self._last_lv_index:
                new = lv.index + 1
            else:
                new = lv.index - 1
            if 0 <= new < len(lv.children) and new in self._lv_map:
                lv.index = new
            return
        self._last_lv_index = lv.index
        fi = self._lv_map[lv.index]
        new_session_idx = self.filtered_items[fi][0]
        if new_session_idx != self._saved_session_idx:
            self._preview_mode = PreviewMode.SUMMARY
        self._saved_session_idx = new_session_idx
        s = self.sessions[new_session_idx]
        ck = self._ops.cache.cache_key(s["file"])
        if self._ops.cache.get(s["session_id"], ck, "stats") is None:
            self._start_task("scan", new_session_idx)
        if self.summaries[new_session_idx] is None:
            self._start_task("summarize", new_session_idx)
        if self._preview_mode == PreviewMode.SUMMARY:
            self._update_preview(fi)

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        if self._selected:
            cmds = []
            for sel_idx in sorted(self._selected):
                s = self.sessions[sel_idx]
                cmd = f"cd {s['project_dir']} && claude --resume {s['session_id']}"
                if self._skip_permissions:
                    cmd += " --dangerously-skip-permissions"
                cmds.append(cmd)
            self.post_message(self.SessionSelected("multi_resume", 0, cmds=cmds))
            return
        fi = self._current_filtered_index()
        if fi is not None:
            idx, session, _ = self.filtered_items[fi]
            cmd = f"cd {session['project_dir']} && claude --resume {session['session_id']}"
            if self._skip_permissions:
                cmd += " --dangerously-skip-permissions"
            self.post_message(self.SessionSelected("select", idx, cmd=cmd))

    # ── Key handling ───────────────────────────────────────

    def _build_resume_cmd(self) -> tuple[int, str] | None:
        fi = self._current_filtered_index()
        if fi is None:
            return None
        idx, session, _ = self.filtered_items[fi]
        cmd = f"cd {session['project_dir']} && claude --resume {session['session_id']}"
        if self._skip_permissions:
            cmd += " --dangerously-skip-permissions"
        return idx, cmd

    def _open_chat(self) -> None:
        """Push the chat screen for conversational session discovery."""
        from ..chat_agent import ChatAgent
        from .chat_screen import ChatScreen

        agent = ChatAgent(self.sessions, self.summaries, self._ops.cache)
        self.app.push_screen(ChatScreen(agent, skip_permissions=self._skip_permissions))

    def on_key(self, event: events.Key) -> None:
        """Handle session-picker key bindings. Unhandled keys bubble to App."""
        search = self.query_one("#sp-search", SearchInput)
        lv = self.query_one("#sp-session-list", ListView)
        scroll_pane = self.query_one("#sp-preview-scroll", VerticalScroll)
        in_search = search == self.screen.focused

        # Arrow keys in search → move to list
        if event.key in ("down", "up") and in_search:
            lv.focus()
            if event.key == "down" and lv.index is not None and lv.index < len(lv.children) - 1:
                lv.index += 1
            elif event.key == "up" and lv.index is not None and lv.index > 0:
                lv.index -= 1
            event.prevent_default()
            event.stop()
            return

        if in_search:
            return  # Let search handle it; unhandled keys bubble to app

        # ── Preview pane mode ──
        if self._in_preview:
            if event.key in ("left", "escape"):
                self._in_preview = False
                scroll_pane.remove_class("focused")
                lv.focus()
                event.prevent_default()
                event.stop()
            elif event.key == "down":
                scroll_pane.scroll_relative(y=10, animate=False)
                event.prevent_default()
                event.stop()
            elif event.key == "up":
                scroll_pane.scroll_relative(y=-10, animate=False)
                event.prevent_default()
                event.stop()
            elif event.key == "enter":
                result = self._build_resume_cmd()
                if result:
                    idx, cmd = result
                    self.post_message(self.SessionSelected("select", idx, cmd=cmd))
                event.prevent_default()
                event.stop()
            elif event.character == "r":
                result = self._build_resume_cmd()
                if result:
                    idx, cmd = result
                    self.post_message(self.SessionSelected("resume", idx, cmd=cmd))
                event.prevent_default()
                event.stop()
            # q, escape etc. — don't stop, let them bubble to app
            return

        # ── List mode ──
        if event.key == "right":
            self._in_preview = True
            scroll_pane.add_class("focused")
            event.prevent_default()
            event.stop()

        elif event.key == "slash":
            fi = self._current_filtered_index()
            if fi is not None:
                self._saved_session_idx = self.filtered_items[fi][0]
            search.focus()
            event.prevent_default()
            event.stop()

        elif event.character == "d":
            self._skip_permissions = not self._skip_permissions
            fi = self._current_filtered_index()
            if fi is not None:
                self._update_preview(fi)
            event.prevent_default()
            event.stop()

        elif event.character == "D":
            fi = self._current_filtered_index()
            if fi is not None:
                idx, session, _ = self.filtered_items[fi]
                ck = self._ops.cache.cache_key(session["file"])
                if self._ops.cache.get(session["session_id"], ck, "deep_summary"):
                    self._update_preview(fi)
                else:
                    self._start_task("deep", idx)
                    self._update_preview(fi)
            event.prevent_default()
            event.stop()

        elif event.character == "p":
            fi = self._current_filtered_index()
            if fi is not None:
                idx, session, _ = self.filtered_items[fi]
                if self._preview_mode == PreviewMode.PATTERNS:
                    self._preview_mode = PreviewMode.SUMMARY
                    self._update_preview(fi)
                else:
                    ck = self._ops.cache.cache_key(session["file"])
                    cached = self._ops.cache.get(session["session_id"], ck, "patterns")
                    self._preview_mode = PreviewMode.PATTERNS
                    if cached:
                        self._display_patterns(idx, cached)
                    else:
                        self.query_one("#sp-preview", Static).update(
                            "[bold yellow]⟳ Analyzing patterns...[/]"
                        )
                        self._start_task("patterns", idx)
            event.prevent_default()
            event.stop()

        elif event.character == "r":
            if self._selected:
                cmds = []
                for sel_idx in sorted(self._selected):
                    s = self.sessions[sel_idx]
                    cmd = f"cd {s['project_dir']} && claude --resume {s['session_id']}"
                    if self._skip_permissions:
                        cmd += " --dangerously-skip-permissions"
                    cmds.append(cmd)
                self.post_message(self.SessionSelected("multi_resume", 0, cmds=cmds))
            else:
                result = self._build_resume_cmd()
                if result:
                    idx, cmd = result
                    self.post_message(self.SessionSelected("resume", idx, cmd=cmd))
            event.prevent_default()
            event.stop()

        elif event.key == "space":
            fi = self._current_filtered_index()
            if fi is not None:
                idx = self.filtered_items[fi][0]
                if idx in self._selected:
                    self._selected.discard(idx)
                else:
                    self._selected.add(idx)
                self._refresh_list()
                n = len(self._selected)
                if hasattr(self.app, "sub_title"):
                    self.app.sub_title = (
                        f"{n} selected — Enter/r to open all" if n
                        else self._picker_title
                    )
            event.prevent_default()
            event.stop()

        elif event.character == "x":
            fi = self._current_filtered_index()
            if fi is not None:
                idx, session, summary = self.filtered_items[fi]
                cache = self._ops.cache
                ck = cache.cache_key(session["file"])
                deep = cache.get(session["session_id"], ck, "deep_summary")
                md = export_context_md(session, summary, deep)
                import subprocess
                subprocess.run(["pbcopy"], input=md.encode(), check=True)
                self._safe_preview_update(
                    f"[bold green]Copied context briefing to clipboard[/]\n\n"
                    f"[dim]{len(md)} characters of markdown[/]\n\n"
                    f"───\n\n{esc(md)}"
                )
            event.prevent_default()
            event.stop()

        elif event.character == "b":
            self._show_bots = not self._show_bots
            self._refresh_list()
            event.prevent_default()
            event.stop()

        elif event.character == "C":
            self._open_chat()
            event.prevent_default()
            event.stop()

        # Keys not handled here (q, escape, 1-4, tab, etc.) bubble to App
