"""Full JSONL session parsing for summaries and context extraction.

Unlike quick_scan (which extracts features for classification), parse_session
reads the full JSONL and extracts user messages, assistant texts, tool names,
and searchable text for summary generation and TUI preview.
"""

import json
from datetime import datetime
from pathlib import Path

from .classify import classify_session
from .display import format_duration


def parse_session(session_file: Path, deep: bool = False) -> tuple[dict, str]:
    """Parse JSONL once — returns (context_dict, searchable_text).

    context_dict includes a 'stats' sub-dict with message counts,
    tool use counts, duration, and classification.
    """
    user_messages = []
    assistant_texts = []
    tool_names = []
    search_parts = []
    total_lines = 0

    assistant_count = 0
    tool_result_count = 0
    system_count = 0
    summary_count = 0
    progress_count = 0
    timestamps = []

    try:
        with open(session_file) as fh:
            for line_num, line in enumerate(fh):
                total_lines = line_num
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = obj.get("type", "")

                ts = obj.get("timestamp")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamps.append(dt.timestamp())
                    except (ValueError, TypeError):
                        pass

                if entry_type == "user":
                    msg = obj.get("message", {}).get("content", "")
                    text = ""
                    if isinstance(msg, str):
                        text = msg
                        search_parts.append(msg)
                    elif isinstance(msg, list):
                        for c in msg:
                            if isinstance(c, dict) and c.get("type") == "text":
                                t = c.get("text", "")
                                search_parts.append(t)
                                text = t
                                break
                    text = text.strip()
                    if text and not text.startswith("[Request interrupted"):
                        limit = 1000 if deep else 500
                        user_messages.append(text[:limit])

                elif entry_type == "assistant":
                    assistant_count += 1
                    amsg = obj.get("message", {})
                    if isinstance(amsg, dict):
                        content = amsg.get("content", [])
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "tool_use":
                                    name = c.get("name", "")
                                    tool_names.append(name)
                                    search_parts.append(name)
                                    inp = c.get("input", {})
                                    if isinstance(inp, dict):
                                        for v in inp.values():
                                            if isinstance(v, str):
                                                search_parts.append(v)
                                elif isinstance(c, dict) and c.get("type") == "text":
                                    t = c.get("text", "")
                                    search_parts.append(t)
                                    limit = 500 if deep else 300
                                    assistant_texts.append(t[:limit])

                elif entry_type == "tool_result":
                    tool_result_count += 1
                    content = obj.get("content", "")
                    if isinstance(content, str):
                        search_parts.append(content)
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                search_parts.append(c.get("text", ""))

                elif entry_type == "system":
                    system_count += 1
                elif entry_type == "summary":
                    summary_count += 1
                elif entry_type == "progress":
                    progress_count += 1
    except Exception:
        pass

    if deep:
        first_msgs = user_messages[:4]
        last_msgs = user_messages[-8:] if len(user_messages) > 4 else []
        first_asst = assistant_texts[:3]
        last_asst = assistant_texts[-5:] if len(assistant_texts) > 3 else []
    else:
        first_msgs = user_messages[:2]
        last_msgs = user_messages[-6:] if len(user_messages) > 2 else []
        first_asst = []
        last_asst = assistant_texts[-3:] if len(assistant_texts) > 0 else []

    duration = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0

    stats = {
        "user_messages": len(user_messages),
        "assistant_messages": assistant_count,
        "tool_uses": len(tool_names),
        "tool_results": tool_result_count,
        "system_entries": system_count,
        "summary_entries": summary_count,
        "progress_entries": progress_count,
        "total_lines": total_lines,
        "duration_secs": round(duration, 1),
        "duration_fmt": format_duration(duration),
        "has_progress": progress_count > 0,
    }
    stats["classification"] = classify_session(stats)

    context = {
        "first_messages": first_msgs,
        "last_messages": last_msgs,
        "first_assistant": first_asst,
        "last_assistant": last_asst,
        "recent_tools": tool_names[-15:],
        "all_tools": list(set(tool_names)),
        "total_user_messages": len(user_messages),
        "total_lines": total_lines,
        "stats": stats,
    }

    search_text = " ".join(search_parts)
    return context, search_text
