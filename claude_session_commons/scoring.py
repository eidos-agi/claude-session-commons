"""Interruption scoring for session prioritization.

Scores how interrupted a session looks (0-100). Higher scores mean
the session is more likely to need resuming. Used by the TUI to
sort and highlight sessions.
"""

import time


def interruption_score(session: dict) -> float:
    """Score how interrupted a session looks (0-100). Higher = more urgent.

    Uses only data available at discovery time (no full parse needed):
    - last_entry_type: what was the session doing when it stopped?
    - recency: how recently was it active?
    - file_size: proxy for invested work
    """
    score = 0.0
    now = time.time()

    last_type = session.get("last_entry_type", "")
    if last_type == "user":
        score += 40  # Typed something, never got a response
    elif last_type == "progress":
        score += 35  # Mid-tool-execution
    elif last_type == "tool_result":
        score += 30  # Tool finished but assistant never responded
    elif last_type == "assistant":
        score += 15  # Normal ending, but could have been mid-thought
    elif last_type == "summary":
        score += 5   # Context compaction — long session, probably fine

    age_hours = (now - session["mtime"]) / 3600
    if age_hours < 0.5:
        score += 30
    elif age_hours < 1:
        score += 25
    elif age_hours < 2:
        score += 20
    elif age_hours < 4:
        score += 15
    elif age_hours < 8:
        score += 10
    elif age_hours < 24:
        score += 5

    size_mb = session["size"] / (1024 * 1024)
    if size_mb > 10:
        score += 15
    elif size_mb > 5:
        score += 10
    elif size_mb > 1:
        score += 5

    return min(score, 100)
