"""Shared Textual widgets for session picking."""

from enum import Enum, auto

from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Input, ListItem, Static

from ..display import lifecycle_badge, relative_time
from ..paths import shorten_path


class PreviewMode(Enum):
    SUMMARY = auto()
    PATTERNS = auto()


class SearchInput(Input):
    """Custom Input that emits Escaped message instead of letting Textual handle it."""

    class Escaped(Message):
        pass

    def key_escape(self) -> None:
        self.value = ""
        self.post_message(self.Escaped())


class DateHeader(ListItem):
    """Non-interactive date group separator."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label_text = label

    def compose(self) -> ComposeResult:
        yield Static(f"[bold dim]── {self.label_text} ──[/]")


class SessionItem(ListItem):
    """A single session row in the list.

    Renders a two-line display with title, badges, path, and age.
    Supports summary text, deep-dive badge, multi-select check, and spinner.
    """
    def __init__(self, idx: int, session: dict, summary: dict | None,
                 has_deep: bool = False, selected: bool = False,
                 is_summarizing: bool = False,
                 bookmark: dict | None = None) -> None:
        super().__init__()
        self.idx = idx
        self.session = session
        self.summary = summary
        self.has_deep = has_deep
        self.selected = selected
        self.is_summarizing = is_summarizing
        self.bookmark = bookmark

    def compose(self) -> ComposeResult:
        if self.summary:
            title = self.summary.get("title", "Unknown")
        elif self.is_summarizing:
            title = "Summarizing..."
        else:
            title = "Queued..."
        short_path = shorten_path(self.session["project_dir"])
        age = relative_time(self.session["mtime"])
        badge = " [bold magenta]◆[/]" if self.has_deep else ""
        check = "[bold green]✓[/] " if self.selected else "  "
        lc_badge = lifecycle_badge(self.bookmark)
        lc_suffix = f" {lc_badge}" if lc_badge else ""
        if self.summary:
            yield Static(f"{check}[bold]{title}[/]{badge}{lc_suffix}\n  [cyan]{short_path}[/]  [dim]{age}[/]")
        elif self.is_summarizing:
            yield Static(f"{check}[bold yellow]{title}[/] [dim]⟳[/]{lc_suffix}\n  [cyan]{short_path}[/]  [dim]{age}[/]")
        else:
            yield Static(f"{check}[dim]{title}[/]{lc_suffix}\n  [cyan]{short_path}[/]  [dim]{age}[/]")


class TaskDone(Message):
    """Single message type for all background task completions."""
    def __init__(self, kind: str, idx: int, result=None, error: str | None = None) -> None:
        super().__init__()
        self.kind = kind
        self.idx = idx
        self.result = result
        self.error = error
