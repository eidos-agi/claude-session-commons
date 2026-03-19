"""Read and analyze past chat logs from ~/.claude/chat-logs/.

NOTE: Legacy — superseded by web dashboard RAG interface.
Still used by ChatAgent (chat_agent.py). Do not extend.

Each log is a JSONL file with events: start, user, assistant, resume, end.
This module aggregates them into structured data the ChatAgent can inject
into its system prompt — no re-running conversations needed.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path.home() / ".claude" / "chat-logs"


@dataclass
class ChatConversation:
    """A single past chat conversation."""

    file: Path
    started: datetime
    turns: int = 0
    queries: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    resumes: list[dict] = field(default_factory=list)  # {num, session_id, title, project_dir}
    search_text: str = ""  # flattened searchable content
    avg_latency: float = 0.0

    @property
    def age_label(self) -> str:
        delta = datetime.now(timezone.utc) - self.started
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)}m ago"
        if hours < 24:
            return f"{int(hours)}h ago"
        days = int(hours / 24)
        return f"{days}d ago"


def load_conversation(path: Path) -> ChatConversation | None:
    """Parse a single JSONL chat log into a ChatConversation."""
    try:
        lines = path.read_text().strip().splitlines()
    except OSError:
        return None
    if not lines:
        return None

    queries: list[str] = []
    responses: list[str] = []
    resumes: list[dict] = []
    latencies: list[float] = []
    search_parts: list[str] = []
    started = None

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts_str = entry.get("ts")
        if ts_str and started is None:
            try:
                started = datetime.fromisoformat(ts_str)
            except ValueError:
                pass

        event = entry.get("event")
        if event == "user":
            text = entry.get("text", "")
            queries.append(text)
            search_parts.append(text)
        elif event == "assistant":
            text = entry.get("text", "")
            responses.append(text)
            search_parts.append(text)
            lat = entry.get("latency_s")
            if lat is not None:
                latencies.append(float(lat))
        elif event == "resume":
            r = {k: entry[k] for k in ("num", "session_id", "title", "project_dir")
                 if k in entry}
            resumes.append(r)
            search_parts.append(r.get("title", ""))
            search_parts.append(r.get("project_dir", ""))

    if started is None:
        stem = path.stem  # e.g. 2026-02-12T20-19-28
        try:
            started = datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
        except ValueError:
            started = datetime.now(timezone.utc)

    return ChatConversation(
        file=path,
        started=started,
        turns=len(queries),
        queries=queries,
        responses=responses,
        resumes=resumes,
        search_text=" ".join(search_parts).lower(),
        avg_latency=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    )


def load_recent(max_conversations: int = 20) -> list[ChatConversation]:
    """Load recent chat conversations, newest first."""
    if not LOG_DIR.exists():
        return []
    files = sorted(LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    convos: list[ChatConversation] = []
    for f in files[:max_conversations]:
        c = load_conversation(f)
        if c and c.turns > 0:
            convos.append(c)
    return convos


def format_history_summary(convos: list[ChatConversation], max_items: int = 10) -> str:
    """Format past conversations as compact text for the system prompt.

    Keeps it tight — the LLM sees what was asked and what was opened,
    not the full responses.
    """
    if not convos:
        return ""

    lines: list[str] = []
    for c in convos[:max_items]:
        q_preview = "; ".join(q[:60] for q in c.queries[:3])
        if len(c.queries) > 3:
            q_preview += f" (+{len(c.queries) - 3} more)"
        resumed = ""
        if c.resumes:
            titles = [r.get("title", "?")[:30] for r in c.resumes]
            resumed = f" -> opened: {', '.join(titles)}"
        lines.append(f"  {c.age_label}: {q_preview}{resumed}")

    return "\n".join(lines)


def search_chats(convos: list[ChatConversation], query: str) -> list[ChatConversation]:
    """Keyword search across past chat conversations.

    Returns matching conversations ranked by hit count, best first.
    """
    import re as _re
    stop = {"the", "and", "for", "was", "that", "with", "from", "this",
            "about", "what", "where", "when", "how", "which", "find",
            "show", "list", "all", "any", "more", "last", "did"}
    words = [w for w in _re.findall(r"[a-z0-9]+", query.lower())
             if len(w) >= 3 and w not in stop]
    if not words:
        return []

    scored: list[tuple[int, ChatConversation]] = []
    for c in convos:
        hits = sum(1 for w in words if w in c.search_text)
        if hits > 0:
            scored.append((hits, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:5]]


def format_chat_matches(matches: list[ChatConversation]) -> str:
    """Format matching past chats for injection into the LLM prompt.

    Shows what was asked, what the LLM found, and what was opened.
    Compact — the LLM already did the analysis, just replay the highlights.
    """
    if not matches:
        return ""
    lines: list[str] = []
    for c in matches:
        lines.append(f"  Chat from {c.age_label} ({c.turns} turns):")
        for i, (q, r) in enumerate(zip(c.queries, c.responses)):
            if i >= 3:
                lines.append(f"    ... +{len(c.queries) - 3} more exchanges")
                break
            lines.append(f"    Q: {q[:120]}")
            # Show first ~2 lines of response
            r_lines = r.strip().split("\n")
            preview = r_lines[0][:120]
            if len(r_lines) > 1:
                preview += f" (+{len(r_lines)-1} lines)"
            lines.append(f"    A: {preview}")
        if c.resumes:
            titles = [r.get("title", "?")[:40] for r in c.resumes]
            lines.append(f"    Opened: {', '.join(titles)}")
    return "\n".join(lines)


def search_queries(convos: list[ChatConversation]) -> list[str]:
    """Extract all unique user queries across conversations."""
    seen: set[str] = set()
    result: list[str] = []
    for c in convos:
        for q in c.queries:
            q_lower = q.strip().lower()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                result.append(q.strip())
    return result


def resumed_sessions(convos: list[ChatConversation]) -> list[dict]:
    """Extract all sessions that were actually opened from chat."""
    seen: set[str] = set()
    result: list[dict] = []
    for c in convos:
        for r in c.resumes:
            sid = r.get("session_id", "")
            if sid and sid not in seen:
                seen.add(sid)
                result.append(r)
    return result
