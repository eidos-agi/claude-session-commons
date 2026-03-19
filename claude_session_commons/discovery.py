"""Session discovery — scan ~/.claude/projects/ for JSONL sessions.

Finds all Claude Code session files, extracts metadata from the tail,
and returns them sorted by last activity. This is the shared discovery
layer used by both claude-boss and claude-resume.
"""

import time
from pathlib import Path

from .paths import decode_project_path
from .tail import get_tail_info

PROJECTS_DIR = Path.home() / ".claude" / "projects"
MIN_SESSION_BYTES_DEFAULT = 100
MAX_SESSIONS_DEFAULT = 200


def find_all_sessions(
    *,
    min_bytes: int = MIN_SESSION_BYTES_DEFAULT,
    projects_dir: Path | None = None,
) -> list[dict]:
    """Find ALL .jsonl session files, sorted by last activity descending.

    Returns list of dicts:
        file: Path — the JSONL file
        session_id: str — filename stem (UUID)
        project_dir: str — decoded project directory path
        mtime: float — epoch time of last activity (from JSONL tail or file stat)
        size: int — file size in bytes
        last_entry_type: str | None — type of last JSONL entry
    """
    root = projects_dir or PROJECTS_DIR
    if not root.exists():
        return []

    sessions = []
    _path_cache: dict[str, str] = {}

    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue

        encoded_name = project_dir.name
        if encoded_name in _path_cache:
            original_path = _path_cache[encoded_name]
        else:
            original_path = decode_project_path(encoded_name)
            _path_cache[encoded_name] = original_path

        for jsonl_file in project_dir.glob("*.jsonl"):
            try:
                stat = jsonl_file.stat()
            except OSError:
                continue
            if stat.st_size < min_bytes:
                continue

            tail = get_tail_info(jsonl_file)
            mtime = tail["timestamp"] if tail["timestamp"] else stat.st_mtime

            sessions.append({
                "file": jsonl_file,
                "session_id": jsonl_file.stem,
                "project_dir": original_path,
                "mtime": mtime,
                "size": stat.st_size,
                "last_entry_type": tail["last_entry_type"],
            })

    sessions.sort(key=lambda s: s["mtime"], reverse=True)
    return sessions


def find_recent_sessions(
    hours: float,
    *,
    max_sessions: int = MAX_SESSIONS_DEFAULT,
    min_bytes: int = MIN_SESSION_BYTES_DEFAULT,
    projects_dir: Path | None = None,
) -> list[dict]:
    """Find sessions modified within the lookback window.

    Pass max_sessions=0 for unlimited (used by daemon/backfill).
    """
    cutoff = time.time() - (hours * 3600)
    all_sessions = find_all_sessions(min_bytes=min_bytes, projects_dir=projects_dir)
    recent = [s for s in all_sessions if s["mtime"] >= cutoff]
    if max_sessions > 0:
        return recent[:max_sessions]
    return recent
