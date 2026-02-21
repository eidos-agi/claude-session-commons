"""SessionOps — callback bundle for TUI operations."""

from dataclasses import dataclass
from typing import Callable

from ..cache import SessionCache


@dataclass
class SessionOps:
    """Everything the session picker TUI needs — one object instead of 7 callbacks."""
    cache: SessionCache
    parse_session: Callable
    get_git_context: Callable
    summarize_quick: Callable
    summarize_deep: Callable
    analyze_patterns: Callable
