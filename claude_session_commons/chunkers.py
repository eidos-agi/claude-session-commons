"""Chunking pipeline for session transcript intelligence.

Parses JSONL session files into embeddable chunks:
- TurnChunker: pairs user/assistant messages by parentUuid
- SubagentChunker: groups progress entries by slug, summarizes via claude -p

See planning/SPEC.md sections 2.1-2.2 for algorithm details.
"""

import dataclasses
import json
import subprocess
from datetime import datetime
from pathlib import Path


@dataclasses.dataclass
class TurnChunk:
    """A user-assistant turn pair, ready for embedding."""
    user_uuid: str
    assistant_uuid: str
    content: str        # The text to embed
    metadata: dict
    timestamp: str      # ISO 8601 from user message


@dataclasses.dataclass
class SubagentChunk:
    """A summarized subagent's work, ready for embedding."""
    slug: str
    content: str        # LLM summary of the subagent's work
    metadata: dict
    timestamp: str      # ISO 8601 from first progress entry


# ── Turn Chunker ──────────────────────────────────────────────

MAX_CONTENT_CHARS = 2000
MAX_TOOL_INPUT_KEYS = 5


def _extract_text(content) -> str:
    """Extract text from a message content field (string or content array)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "").strip()
    return ""


def _extract_assistant_text(content: list) -> str:
    """Extract all text blocks from assistant message content array."""
    parts = []
    if not isinstance(content, list):
        return ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _extract_tools(content: list) -> tuple[list[str], list[str]]:
    """Extract tool names and compact tool call descriptions from content.

    Returns (tool_names, tool_descriptions) where descriptions are like:
        Edit(file_path=/path/to/file.py)
        Bash(command=git status)
    """
    names = []
    descriptions = []
    if not isinstance(content, list):
        return names, descriptions
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        if not name:
            continue
        names.append(name)
        inp = block.get("input", {})
        if isinstance(inp, dict):
            # Show key=value for small string values, just key for large ones
            parts = []
            for k, v in list(inp.items())[:MAX_TOOL_INPUT_KEYS]:
                if isinstance(v, str) and len(v) <= 80:
                    parts.append(f"{k}={v}")
                elif isinstance(v, str):
                    parts.append(f"{k}=<{len(v)} chars>")
                else:
                    parts.append(k)
            descriptions.append(f"{name}({', '.join(parts)})")
        else:
            descriptions.append(name)
    return names, descriptions


def _extract_files_touched(content: list) -> list[str]:
    """Extract file paths from Edit/Write/Read tool calls."""
    files = []
    if not isinstance(content, list):
        return files
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        inp = block.get("input", {})
        if not isinstance(inp, dict):
            continue
        if name in ("Edit", "Write", "Read") and "file_path" in inp:
            fp = inp["file_path"]
            if isinstance(fp, str) and fp not in files:
                files.append(fp)
    return files


def _count_tokens(content: list) -> int:
    """Rough token count from assistant content (chars / 4)."""
    total = 0
    if not isinstance(content, list):
        return 0
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                total += len(block.get("text", ""))
            elif block.get("type") == "tool_use":
                total += len(json.dumps(block.get("input", {})))
    return total // 4


def chunk_turns(session_path: str | Path) -> list[TurnChunk]:
    """Parse a JSONL session into user-assistant turn chunks.

    Algorithm:
    1. Read JSONL, build uuid -> entry lookup
    2. For each user entry, find its assistant reply (where assistant.parentUuid == user.uuid)
    3. Concatenate into embeddable content, truncate to 2000 chars
    4. Skip turns with empty assistant responses or only tool results
    """
    session_path = Path(session_path)
    entries_by_uuid: dict[str, dict] = {}
    user_entries: list[dict] = []
    assistant_by_parent: dict[str, dict] = {}

    with open(session_path) as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = obj.get("type", "")
            uuid = obj.get("uuid")

            if uuid:
                entries_by_uuid[uuid] = obj

            if entry_type == "user":
                user_entries.append(obj)
            elif entry_type == "assistant":
                parent = obj.get("parentUuid")
                if parent:
                    assistant_by_parent[parent] = obj

    chunks = []
    for user_obj in user_entries:
        user_uuid = user_obj.get("uuid", "")
        if not user_uuid:
            continue

        assistant_obj = assistant_by_parent.get(user_uuid)
        if not assistant_obj:
            continue

        # Extract user text
        user_text = _extract_text(user_obj.get("message", {}).get("content", ""))
        if not user_text:
            continue

        # Extract assistant content
        a_content = assistant_obj.get("message", {}).get("content", [])
        assistant_text = _extract_assistant_text(a_content)
        tool_names, tool_descs = _extract_tools(a_content)
        files_touched = _extract_files_touched(a_content)

        # Skip turns where assistant only has tool results, no text
        if not assistant_text and not tool_descs:
            continue

        # Build embeddable content
        parts = [f"USER: {user_text}"]
        if assistant_text:
            parts.append(f"\nASSISTANT: {assistant_text}")
        if tool_descs:
            parts.append(f"\nTOOLS: {', '.join(tool_descs)}")

        content = "\n".join(parts)
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS - 14] + "\n[...truncated]"

        # Extract timestamp
        ts = user_obj.get("timestamp", "")
        if not ts:
            try:
                ts = datetime.now().isoformat()
            except Exception:
                ts = ""

        # Model from assistant
        model = assistant_obj.get("message", {}).get("model", "")

        # Token count estimate
        token_count = _count_tokens(a_content)

        metadata = {
            "user_uuid": user_uuid,
            "assistant_uuid": assistant_obj.get("uuid", ""),
            "model": model,
            "tools_used": list(set(tool_names)),
            "files_touched": files_touched,
            "token_count": token_count,
        }

        chunks.append(TurnChunk(
            user_uuid=user_uuid,
            assistant_uuid=assistant_obj.get("uuid", ""),
            content=content,
            metadata=metadata,
            timestamp=ts,
        ))

    return chunks


# ── Subagent Chunker ──────────────────────────────────────────

MIN_PROGRESS_ENTRIES = 5
MAX_SUMMARY_INPUT_CHARS = 4000


def _summarize_subagent(initial_prompt: str, work_text: str) -> str:
    """Summarize a subagent's work via claude -p haiku.

    Returns the summary string, or a fallback if the call fails.
    """
    prompt = f"""Summarize this AI agent's work in 2-3 sentences. Focus on:
what was researched/built, key findings, and outcome.

Initial task: {initial_prompt[:500]}

Work performed:
{work_text[:MAX_SUMMARY_INPUT_CHARS]}"""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--no-session-persistence",
             "--output-format", "json", "--model", "claude-haiku-4-5-20251001"],
            capture_output=True, text=True, timeout=30,
            stdin=subprocess.DEVNULL,
        )
        output = result.stdout.strip()
        parsed = json.loads(output)
        # Extract text from structured output
        if isinstance(parsed, dict):
            if "result" in parsed:
                return str(parsed["result"]).strip()
            if "text" in parsed:
                return str(parsed["text"]).strip()
        return output[:500]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        # Fallback: use the initial prompt as a minimal summary
        return f"Agent task: {initial_prompt[:200]}"


def chunk_subagents(session_path: str | Path, summarize: bool = True) -> list[SubagentChunk]:
    """Parse a JSONL session into subagent summary chunks.

    Algorithm:
    1. Group all type=progress entries by slug
    2. For each slug group with > 5 entries (skip trivial agents):
       a. Extract initial prompt from first entry
       b. Collect text from subsequent entries
       c. Summarize via claude -p haiku (if summarize=True)
    3. Return SubagentChunk list

    Set summarize=False for testing (skips LLM calls, uses raw text).
    """
    session_path = Path(session_path)
    progress_by_slug: dict[str, list[dict]] = {}

    with open(session_path) as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") != "progress":
                continue

            slug = obj.get("slug", "")
            if not slug:
                continue

            if slug not in progress_by_slug:
                progress_by_slug[slug] = []
            progress_by_slug[slug].append(obj)

    chunks = []
    for slug, entries in progress_by_slug.items():
        if len(entries) < MIN_PROGRESS_ENTRIES:
            continue

        # Extract initial prompt from first entry
        first = entries[0]
        initial_prompt = ""
        data = first.get("data", {})
        if isinstance(data, dict):
            initial_prompt = data.get("message", "")
        if isinstance(initial_prompt, list):
            # Sometimes message is a content array
            for block in initial_prompt:
                if isinstance(block, dict) and block.get("type") == "text":
                    initial_prompt = block.get("text", "")
                    break
            else:
                initial_prompt = str(initial_prompt)[:200]
        if not isinstance(initial_prompt, str):
            initial_prompt = str(initial_prompt)[:200]

        # Collect all text from progress entries
        work_parts = []
        tools_used = set()
        for entry in entries[1:]:
            d = entry.get("data", {})
            if isinstance(d, dict):
                msg = d.get("message", "")
                if isinstance(msg, str) and msg.strip():
                    work_parts.append(msg.strip()[:300])
                elif isinstance(msg, list):
                    for block in msg:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                work_parts.append(block.get("text", "")[:300])
                            elif block.get("type") == "tool_use":
                                tools_used.add(block.get("name", ""))

        work_text = "\n".join(work_parts)

        # Summarize or use raw text
        if summarize and work_text:
            content = _summarize_subagent(initial_prompt, work_text)
        elif work_text:
            content = f"Agent task: {initial_prompt[:200]}\n\nWork: {work_text[:1800]}"
        else:
            content = f"Agent task: {initial_prompt[:500]}"

        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS - 14] + "\n[...truncated]"

        # Timestamp from first entry
        ts = first.get("timestamp", "")
        if not ts:
            ts = datetime.now().isoformat()

        metadata = {
            "slug": slug,
            "initial_prompt_preview": initial_prompt[:200],
            "tools_used": sorted(tools_used),
            "progress_line_count": len(entries),
        }

        chunks.append(SubagentChunk(
            slug=slug,
            content=content,
            metadata=metadata,
            timestamp=ts,
        ))

    return chunks
