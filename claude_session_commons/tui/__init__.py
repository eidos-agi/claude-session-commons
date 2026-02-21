"""Shared Textual TUI components for session picking.

Public API:
    # Widgets
    SearchInput — Input that emits Escaped on Esc
    DateHeader — Non-interactive date group separator
    SessionItem — Two-line session row with summary, badges, spinner
    TaskDone — Background task completion message
    PreviewMode — Enum for preview pane mode (SUMMARY, PATTERNS)

    # Dataclass
    SessionOps — Callback bundle for TUI operations

    # Main widget
    SessionPickerPanel — Embeddable session picker with full functionality
"""

from .widgets import DateHeader, PreviewMode, SearchInput, SessionItem, TaskDone
from .ops import SessionOps
from .session_picker import SessionPickerPanel

__all__ = [
    "DateHeader",
    "PreviewMode",
    "SearchInput",
    "SessionItem",
    "SessionOps",
    "SessionPickerPanel",
    "TaskDone",
]
