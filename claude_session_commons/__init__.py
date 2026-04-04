"""claude-session-commons — Shared session discovery and classification.

Public API:
    # Discovery
    find_all_sessions(*, min_bytes, projects_dir)
    find_recent_sessions(hours, *, max_sessions, min_bytes, projects_dir)

    # Caching
    SessionCache(cache_dir)

    # Classification
    quick_scan(session_file) -> dict
    classify_session(stats) -> str
    get_label(session_file, cache) -> str
    get_label_deep(session_file, cache) -> str

    # Parsing
    parse_session(session_file, deep) -> (context_dict, search_text)

    # Display helpers
    relative_time(mtime, *, compact=False) -> str
    get_date_group(mtime) -> str
    format_size(size_bytes) -> str
    format_duration(secs) -> str
    shorten_path(path) -> str

    # Path decoding
    decode_project_path(encoded) -> str

    # Scoring
    interruption_score(session) -> float

    # Git
    get_git_context(project_dir) -> dict
    has_uncommitted_changes(project_dir) -> bool

    # Export
    export_context_md(session, summary, deep) -> str

    # Summarization (AI via claude -p)
    summarize_quick(context, project_dir, git) -> dict
    summarize_deep(context, project_dir, quick_summary, git) -> dict
    analyze_patterns(context, project_dir, summary) -> dict

    # Hierarchical summaries (L2 project, L3 portfolio)
    generate_project_summary(project_path, conn, cache) -> dict
    summarize_portfolio(project_summaries) -> dict

    # Chat agent (legacy — use web dashboard RAG interface instead)
    ChatAgent(sessions, summaries, cache)

    # Chat history (legacy — use web dashboard RAG interface instead)
    load_recent() -> list[ChatConversation]
"""

from .cache import SessionCache
# NOTE: ChatAgent/ChatConversation still importable directly from their modules
# but removed from the public API — superseded by web dashboard RAG interface
from .classify import (
    classify_session,
    get_label,
    get_label_deep,
    quick_scan,
)
from .discovery import find_all_sessions, find_recent_sessions
from .display import format_duration, format_size, get_date_group, lifecycle_badge, relative_time
from .export import export_context_md
from .git_context import get_git_context, has_uncommitted_changes
from .parse import parse_session
from .paths import decode_project_path, shorten_path
from .scoring import interruption_score
from .summarize import analyze_patterns, summarize_deep, summarize_quick
from .tail import get_tail_info

__all__ = [
    "SessionCache",
    "classify_session",
    "decode_project_path",
    "export_context_md",
    "find_all_sessions",
    "find_recent_sessions",
    "format_duration",
    "format_size",
    "get_date_group",
    "get_git_context",
    "get_label",
    "get_label_deep",
    "get_tail_info",
    "has_uncommitted_changes",
    "interruption_score",
    "lifecycle_badge",
    "parse_session",
    "quick_scan",
    "relative_time",
    "shorten_path",
    "summarize_quick",
    "summarize_deep",
    "analyze_patterns",
]
