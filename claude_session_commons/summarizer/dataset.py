"""Step 1 of 4: Generate a training dataset from Claude Code sessions.

Reads raw session JSONL files, extracts time-window slices (5m / 30m / 2h),
and pairs each window with a label produced by the best available summarizer
(cached session_summary output, or falling back to the last user message).

Output: one JSONL file per line:
    {"window_text": "User: ...\nAssistant: ...", "summary": "...", "window": "30m"}

Usage:
    python -m claude_session_commons.summarizer.dataset \
        --sessions ~/.claude/projects \
        --output training.jsonl \
        --n 2000 \
        --window 30m
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterator


# ── Window extraction ──────────────────────────────────────────────────────────

WINDOWS_MINUTES = {"5m": 5, "30m": 30, "2h": 120}


def _iter_messages(filepath: str) -> Iterator[dict]:
    """Yield parsed JSONL lines from a session file."""
    try:
        with open(filepath, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def extract_window(filepath: str, minutes: int) -> str:
    """Return the last `minutes` of conversation from a session file.

    Walks the JSONL tail, collecting user + assistant text until the
    time budget is exhausted (by timestamp) or we hit the start of file.
    Returns a plain-text block:

        User: <message>
        Assistant: <response>
        [Bash] <command>
        ...
    """
    import time as _time

    cutoff = _time.time() - minutes * 60
    lines: list[str] = []

    messages = list(_iter_messages(filepath))

    for msg in messages:
        ts = msg.get("timestamp", 0)
        if ts and ts < cutoff:
            continue

        role = msg.get("role", "")
        content = msg.get("message", {})

        if role == "human":
            # Human turn: collect text + tool_result blocks
            parts = content if isinstance(content, list) else [content]
            for part in parts:
                if isinstance(part, str):
                    lines.append(f"User: {part.strip()}")
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        lines.append(f"User: {part['text'].strip()}")
                    elif part.get("type") == "tool_result":
                        snippet = str(part.get("content", ""))[:200]
                        lines.append(f"[Result] {snippet}")

        elif role == "assistant":
            parts = content if isinstance(content, list) else [content]
            for part in parts:
                if isinstance(part, str):
                    lines.append(f"Assistant: {part.strip()}")
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        text = part["text"].strip()
                        if text:
                            lines.append(f"Assistant: {text[:500]}")
                    elif part.get("type") == "tool_use":
                        tool = part.get("name", "tool")
                        inp = part.get("input", {})
                        if "command" in inp:
                            lines.append(f"[{tool}] {inp['command'][:200]}")
                        elif "file_path" in inp:
                            lines.append(f"[{tool}] {inp['file_path']}")
                        else:
                            lines.append(f"[{tool}]")

    return "\n".join(lines).strip()


# ── Label generation ───────────────────────────────────────────────────────────

def _load_cached_summary(filepath: str) -> str | None:
    """Return a cached session summary if one exists alongside the session file.

    The cache lives at <session_id>.cache.json next to the JSONL, written by
    claude_session_commons.cache.SessionCache.
    """
    cache_path = filepath.replace(".jsonl", ".cache.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        # Cache stores summaries under arbitrary keys — find the first string value
        for v in data.values():
            if isinstance(v, dict):
                summary = v.get("summary", {})
                if isinstance(summary, dict):
                    return summary.get("what_was_done") or summary.get("state")
                if isinstance(summary, str) and len(summary) > 10:
                    return summary
        return None
    except (OSError, json.JSONDecodeError):
        return None


def _fallback_label(window_text: str) -> str:
    """Last-resort label: last user message, trimmed to 500 chars."""
    last = ""
    for line in window_text.splitlines():
        if line.startswith("User:"):
            last = line[5:].strip()
    text = last or window_text.strip()
    if len(text) > 500:
        text = text[:500].rsplit(" ", 1)[0] + "…"
    return text


# ── Dataset generation ─────────────────────────────────────────────────────────

def find_sessions(sessions_dir: str) -> list[str]:
    """Return all JSONL session files under sessions_dir."""
    root = Path(sessions_dir).expanduser()
    return [str(p) for p in root.rglob("*.jsonl") if p.stat().st_size > 1024]


def generate_dataset(
    sessions_dir: str = "~/.claude/projects",
    output: str = "training.jsonl",
    n: int = 2000,
    window: str = "30m",
    seed: int = 42,
) -> int:
    """Generate a training dataset and write it to `output`.

    Returns the number of examples written.

    Args:
        sessions_dir: Root directory containing Claude Code JSONL session files.
        output: Path to write the output JSONL.
        n: Maximum number of examples to generate.
        window: Time window to extract ("5m", "30m", or "2h").
        seed: Random seed for reproducible sampling.
    """
    minutes = WINDOWS_MINUTES.get(window, 30)
    sessions = find_sessions(sessions_dir)
    random.seed(seed)
    random.shuffle(sessions)

    written = 0
    with open(output, "w") as out:
        for filepath in sessions:
            if written >= n:
                break

            window_text = extract_window(filepath, minutes)
            if len(window_text) < 50:
                continue  # too sparse to learn from

            label = _load_cached_summary(filepath) or _fallback_label(window_text)
            if not label or len(label) < 10:
                continue

            out.write(json.dumps({
                "window_text": window_text[:2000],  # cap input length
                "summary": label,
                "window": window,
                "source": os.path.basename(os.path.dirname(filepath)),
            }) + "\n")
            written += 1

    print(f"Wrote {written} examples to {output}")
    return written


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summarization training data from Claude Code sessions")
    parser.add_argument("--sessions", default="~/.claude/projects", help="Root sessions directory")
    parser.add_argument("--output", default="training.jsonl", help="Output JSONL file")
    parser.add_argument("--n", type=int, default=2000, help="Max examples")
    parser.add_argument("--window", default="30m", choices=["5m", "30m", "2h"], help="Time window")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.sessions, args.output, args.n, args.window, args.seed)


if __name__ == "__main__":
    main()
