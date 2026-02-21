"""Conversational chat agent for session discovery.

Uses claude -p (CLI) for all LLM calls. Never imports anthropic SDK.
Two-tier intelligence:
  Tier 1: Top ~50 sessions with metadata baked into the system prompt
  Tier 2: Keyword scan across all sessions' cached search_text
"""

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .cache import SessionCache
from .chat_history import (
    ChatConversation,
    format_chat_matches,
    format_history_summary,
    load_recent,
    search_chats,
)
from .display import relative_time
from .paths import shorten_path


@dataclass
class SessionRef:
    """Compact session reference for the agent's catalog."""

    num: int  # 1-indexed display number
    session_id: str
    project_dir: str
    title: str
    goal: str
    state: str
    age: str
    lifecycle: str
    next_actions: list[str]
    files: list[str]
    original_idx: int  # index into sessions[] list


class ChatAgent:
    """Stateful conversation agent over session data.

    Maintains a numbered catalog of recent sessions and a conversation
    history.  Each call to ``respond()`` builds a prompt with the catalog,
    any deep-search results, and the conversation so far, then shells out
    to ``claude -p`` for an answer.
    """

    MAX_CATALOG = 50
    MAX_HISTORY = 10  # turns (user+assistant pairs)
    LOG_DIR = Path.home() / ".claude" / "chat-logs"

    def __init__(
        self,
        sessions: list[dict],
        summaries: list[dict | None],
        cache: SessionCache,
    ) -> None:
        self._sessions = sessions
        self._summaries = summaries
        self._cache = cache
        self._history: list[tuple[str, str]] = []  # (role, text)
        self._catalog: list[SessionRef] = []
        self._log_file = self._init_log()
        self._past_chats = load_recent(max_conversations=20)
        self._build_catalog()
        n_sessions, n_projects = self.welcome_info()
        self._log("start", n_sessions=n_sessions, n_projects=n_projects,
                  catalog_size=min(len(self._sessions), self.MAX_CATALOG),
                  past_chats=len(self._past_chats))

    # ── Chat log ───────────────────────────────────────────

    def _init_log(self) -> Path:
        """Create a timestamped JSONL log file for this conversation."""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        return self.LOG_DIR / f"{ts}.jsonl"

    def _log(self, event: str, **data) -> None:
        """Append a single JSONL event to the chat log."""
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass  # logging is best-effort

    # ── Catalog ───────────────────────────────────────────

    def _build_catalog(self) -> None:
        """Build numbered SessionRef list from the top N sessions."""
        num = 1
        for i, (s, sm) in enumerate(zip(self._sessions, self._summaries)):
            if num > self.MAX_CATALOG:
                break
            ck = self._cache.cache_key(s["file"])
            deep = self._cache.get(s["session_id"], ck, "deep_summary")
            best = deep or sm
            if best is None:
                # no summary yet — include with minimal info
                title = "Unsummarized session"
                goal = ""
                state = ""
                files = []
            else:
                title = best.get("title", "Unknown")
                goal = best.get("goal", best.get("objective", ""))
                state = best.get("state", "")
                files = best.get("files", [])

            bookmark = self._cache.get(s["session_id"], ck, "bookmark")
            lifecycle = ""
            next_actions: list[str] = []
            if bookmark and isinstance(bookmark, dict):
                lifecycle = bookmark.get("lifecycle_state", "")
                next_actions = bookmark.get("next_actions", [])

            self._catalog.append(
                SessionRef(
                    num=num,
                    session_id=s["session_id"],
                    project_dir=s["project_dir"],
                    title=title,
                    goal=goal,
                    state=state,
                    age=relative_time(s["mtime"], compact=True),
                    lifecycle=lifecycle,
                    next_actions=next_actions,
                    files=files,
                    original_idx=i,
                )
            )
            num += 1

    def _format_catalog(self) -> str:
        """Format catalog as compact text for the system prompt."""
        if not self._catalog:
            return "No sessions available."
        lines: list[str] = []
        for ref in self._catalog:
            short = shorten_path(ref.project_dir)
            lc = f" | {ref.lifecycle}" if ref.lifecycle else ""
            lines.append(f"#{ref.num} | {ref.title} | {short} | {ref.age}{lc}")
            if ref.goal:
                lines.append(f"   Goal: {ref.goal[:200]}")
            if ref.state:
                lines.append(f"   Left off: {ref.state[:200]}")
            if ref.next_actions:
                actions = "; ".join(ref.next_actions[:3])
                lines.append(f"   Next: {actions}")
        return "\n".join(lines)

    def welcome_info(self) -> tuple[int, int]:
        """Return (session_count, project_count) for the welcome message."""
        projects = {s["project_dir"] for s in self._sessions}
        return len(self._sessions), len(projects)

    # ── Deep search ───────────────────────────────────────

    def _deep_search(self, query: str) -> list[SessionRef]:
        """Keyword scan across ALL sessions' search_text.

        Returns SessionRefs for matches NOT already in the catalog.
        """
        # Extract meaningful words (3+ chars, skip common words)
        stop = {"the", "and", "for", "was", "that", "with", "from", "this",
                "about", "what", "where", "when", "how", "which", "find",
                "show", "list", "all", "any", "more", "last", "did"}
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower())
                 if len(w) >= 3 and w not in stop]
        if not words:
            return []

        catalog_ids = {ref.session_id for ref in self._catalog}
        scored: list[tuple[int, int, dict, dict | None]] = []

        for i, s in enumerate(self._sessions):
            if s["session_id"] in catalog_ids:
                continue
            ck = self._cache.cache_key(s["file"])
            search_text = self._cache.get(s["session_id"], ck, "search_text") or ""
            hits = sum(1 for w in words if w in search_text)
            if hits > 0:
                sm = self._summaries[i] if i < len(self._summaries) else None
                scored.append((hits, i, s, sm))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SessionRef] = []
        start_num = len(self._catalog) + 1

        for rank, (_, i, s, sm) in enumerate(scored[:10]):
            best = sm or {}
            ck = self._cache.cache_key(s["file"])
            bookmark = self._cache.get(s["session_id"], ck, "bookmark")

            ref = SessionRef(
                num=start_num + rank,
                session_id=s["session_id"],
                project_dir=s["project_dir"],
                title=best.get("title", "Unsummarized"),
                goal=best.get("goal", ""),
                state=best.get("state", ""),
                age=relative_time(s["mtime"], compact=True),
                lifecycle=bookmark.get("lifecycle_state", "") if bookmark else "",
                next_actions=bookmark.get("next_actions", []) if bookmark else [],
                files=best.get("files", []),
                original_idx=i,
            )
            results.append(ref)
            # Register in catalog so "resume #N" works
            self._catalog.append(ref)

        return results

    def _broader_search(self, query: str) -> list[SessionRef]:
        """Fallback search when _deep_search finds nothing.

        More aggressive: shorter fragments, partial prefix matching,
        direct scan of summary titles/goals, and sessions with no
        search_text cached.
        """
        # Extract ALL words, including short ones
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) >= 2]
        if not words:
            return []

        # Also generate prefixes for longer words (e.g. "graphql" → "graph")
        prefixes = set()
        for w in words:
            if len(w) >= 5:
                prefixes.add(w[:4])
            prefixes.add(w)

        already = {ref.session_id for ref in self._catalog}
        scored: list[tuple[float, int, dict, dict | None]] = []

        for i, s in enumerate(self._sessions):
            if s["session_id"] in already:
                continue

            score = 0.0
            ck = self._cache.cache_key(s["file"])

            # Check search_text with partial matching
            search_text = self._cache.get(s["session_id"], ck, "search_text") or ""
            if search_text:
                for p in prefixes:
                    if p in search_text:
                        score += 1.0

            # Also check summary title and goal directly
            sm = self._summaries[i] if i < len(self._summaries) else None
            if sm:
                title_goal = f"{sm.get('title', '')} {sm.get('goal', '')}".lower()
                for p in prefixes:
                    if p in title_goal:
                        score += 2.0  # title/goal matches are stronger signals

            # Check project directory path
            proj = s["project_dir"].lower()
            for p in prefixes:
                if p in proj:
                    score += 0.5

            if score > 0:
                scored.append((score, i, s, sm))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SessionRef] = []
        start_num = max((ref.num for ref in self._catalog), default=0) + 1

        for rank, (_, i, s, sm) in enumerate(scored[:10]):
            best = sm or {}
            ck = self._cache.cache_key(s["file"])
            bookmark = self._cache.get(s["session_id"], ck, "bookmark")

            ref = SessionRef(
                num=start_num + rank,
                session_id=s["session_id"],
                project_dir=s["project_dir"],
                title=best.get("title", "Unsummarized"),
                goal=best.get("goal", ""),
                state=best.get("state", ""),
                age=relative_time(s["mtime"], compact=True),
                lifecycle=bookmark.get("lifecycle_state", "") if bookmark else "",
                next_actions=bookmark.get("next_actions", []) if bookmark else [],
                files=best.get("files", []),
                original_idx=i,
            )
            results.append(ref)
            self._catalog.append(ref)

        return results

    def _count_unindexed(self) -> int:
        """Count sessions without cached search_text."""
        count = 0
        for s in self._sessions:
            ck = self._cache.cache_key(s["file"])
            if not self._cache.get(s["session_id"], ck, "search_text"):
                count += 1
        return count

    def _format_refs(self, refs: list[SessionRef]) -> str:
        lines: list[str] = []
        for ref in refs:
            short = shorten_path(ref.project_dir)
            lc = f" | {ref.lifecycle}" if ref.lifecycle else ""
            lines.append(f"#{ref.num} | {ref.title} | {short} | {ref.age}{lc}")
            if ref.goal:
                lines.append(f"   Goal: {ref.goal[:200]}")
            if ref.state:
                lines.append(f"   Left off: {ref.state[:200]}")
        return "\n".join(lines)

    # ── Prompt building ───────────────────────────────────

    def _system_prompt(self) -> str:
        n_sessions, n_projects = self.welcome_info()
        prompt = f"""You are a session navigator inside claude-resume, a terminal tool for finding and resuming Claude Code sessions.

BEHAVIOR:
- Be concise. Responses should fit a terminal (under 20 lines typically).
- Reference sessions by their #N number.
- When listing sessions, include #N, title, project, age on one line.
- When asked for details, include goal, state, files, bookmark data.
- Be decisive — if one session clearly matches, say so.
- Use plain text. No markdown headers or bullet points with *.
- Never ask unnecessary questions. If the user's intent is clear, act on it.
- If the user wants to resume a session, tell them to type the number (e.g. just "7") to open it.
- You have access to past chat history and past chat search matches. Use them to understand what the user has been searching for, which sessions they opened before, and patterns in their work. If a past chat already found what the user is looking for, reference the previous answer — don't make them re-discover it. Say things like "You asked about this 2d ago and opened session X" when relevant.

SESSION CATALOG ({n_sessions} sessions, {n_projects} projects):
{self._format_catalog()}"""

        chat_summary = format_history_summary(self._past_chats)
        if chat_summary:
            prompt += f"\n\nPAST CHAT HISTORY (recent searches and sessions opened):\n{chat_summary}"

        return prompt

    def _format_history(self) -> str:
        if not self._history:
            return ""
        lines: list[str] = []
        for role, text in self._history:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    def _build_prompt(self, user_input: str, extra_refs: list[SessionRef] | None = None,
                      unindexed: int = 0,
                      chat_matches: list[ChatConversation] | None = None) -> str:
        prompt = self._system_prompt()
        if extra_refs:
            prompt += f"\n\nADDITIONAL MATCHES (from archive search):\n{self._format_refs(extra_refs)}"
        if chat_matches:
            prompt += f"\n\nPAST CHAT MATCHES (previous searches relevant to this query):\n{format_chat_matches(chat_matches)}"
        if unindexed > 0:
            prompt += f"\n\nNOTE: {unindexed} older sessions have not been indexed yet and may not appear in search results."
        history = self._format_history()
        if history:
            prompt += f"\n\nCONVERSATION:\n{history}"
        prompt += f"\n\nUser: {user_input}"
        return prompt

    # ── LLM call ──────────────────────────────────────────

    def _call_claude(self, prompt: str, timeout: int = 30) -> str:
        """Call claude -p and return a text response."""
        try:
            cmd = [
                "claude", "-p", prompt,
                "--no-session-persistence",
                "--output-format", "json",
                "--model", "claude-haiku-4-5-20251001",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, stdin=subprocess.DEVNULL,
            )
            output = result.stdout.strip()
            try:
                parsed = json.loads(output)
                if isinstance(parsed.get("result"), str):
                    return parsed["result"]
                if isinstance(parsed.get("structured_output"), str):
                    return parsed["structured_output"]
                return output
            except json.JSONDecodeError:
                return output if output else "No response received. Try again."
        except subprocess.TimeoutExpired:
            return "Response timed out. Try a simpler question or use 'resume #N' directly."
        except FileNotFoundError:
            return "Claude CLI not found. Make sure 'claude' is installed and in your PATH."
        except Exception as e:
            return f"Error: {e}"

    # ── Public API ────────────────────────────────────────

    def respond(self, user_input: str) -> str:
        """Send a message and get a response. Blocking — call from a thread."""
        self._log("user", text=user_input)

        # Tier 2: keyword scan across all sessions
        extra = self._deep_search(user_input)
        # Tier 3: broader search if keyword scan found nothing
        if not extra:
            extra = self._broader_search(user_input)

        # Also search past C chats — the user's own search history is signal
        chat_matches = search_chats(self._past_chats, user_input)

        search_refs = [r.num for r in extra] if extra else []
        # Note unindexed sessions so the LLM can mention it
        unindexed = self._count_unindexed()
        prompt = self._build_prompt(user_input, extra or None, unindexed=unindexed,
                                    chat_matches=chat_matches or None)

        t0 = time.monotonic()
        response = self._call_claude(prompt)
        elapsed = round(time.monotonic() - t0, 2)

        self._history.append(("user", user_input))
        self._history.append(("assistant", response))

        # Extract #N refs from response for the log
        resp_refs = [int(m.group(1)) for m in re.finditer(r"#(\d+)", response)]

        self._log("assistant", text=response, refs=resp_refs,
                  search_hits=search_refs, chat_hits=len(chat_matches),
                  latency_s=elapsed)

        # Trim history to keep prompt size manageable
        max_entries = self.MAX_HISTORY * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

        return response

    def parse_resume_request(self, text: str) -> int | None:
        """Parse resume commands. Returns #N or None.

        Accepts: "resume #7", "open #3", "resume 7", "#7", "7"
        A bare number only matches if it's the entire input.
        """
        text = text.strip()
        # Explicit command: "resume #7", "open 3"
        m = re.match(r"(?:resume|open)\s+#?(\d+)", text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # Bare reference: "#7" or just "7"
        m = re.match(r"^#?(\d+)$", text)
        if m:
            return int(m.group(1))
        return None

    def get_session_by_num(self, num: int) -> dict | None:
        """Look up original session dict by catalog #N."""
        for ref in self._catalog:
            if ref.num == num:
                return self._sessions[ref.original_idx]
        return None

    def get_ref_by_num(self, num: int) -> SessionRef | None:
        """Look up SessionRef by catalog #N."""
        for ref in self._catalog:
            if ref.num == num:
                return ref
        return None

    def log_resume(self, num: int) -> None:
        """Log a session resume event."""
        ref = self.get_ref_by_num(num)
        if ref:
            self._log("resume", num=num, session_id=ref.session_id,
                      project_dir=ref.project_dir, title=ref.title)

    def log_end(self) -> None:
        """Log end of chat conversation."""
        self._log("end", turns=len(self._history) // 2)

    def build_resume_cmd(self, num: int, skip_permissions: bool = True) -> str | None:
        """Build the shell command to resume session #N."""
        ref = self.get_ref_by_num(num)
        if ref is None:
            return None
        cmd = f"cd {ref.project_dir} && claude --resume {ref.session_id}"
        if skip_permissions:
            cmd += " --dangerously-skip-permissions"
        return cmd
