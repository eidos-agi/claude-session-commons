"""Mtime-based session cache.

One JSON file per session, keyed on cache_key = md5(path:mtime).
When the JSONL file changes, the cache key invalidates and stale
fields are ignored on next read.

Fields independent of cache_key (persist across changes):
    last_seen — timestamp of last TUI view (for cooldown/dedup)
    bookmark  — human-authored lifecycle signal (from /bookmark skill)
"""

import hashlib
import json
import time
from pathlib import Path

CLAUDE_DIR = Path.home() / ".claude"
COOLDOWN_SECONDS = 600


class SessionCache:
    """Unified session cache for any Claude Code tool.

    Default cache dir: ~/.claude/session-cache/
    Callers can override (e.g. ~/.claude/resume-summaries/ for backward compat).
    """

    def __init__(self, cache_dir: Path | None = None):
        self._dir = cache_dir or (CLAUDE_DIR / "session-cache")
        self._dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, session_file: Path) -> str:
        """Create invalidation key from path + mtime."""
        try:
            mtime = session_file.stat().st_mtime
        except OSError:
            mtime = 0
        return hashlib.md5(f"{session_file}:{mtime}".encode()).hexdigest()

    # Fields that persist across cache_key changes (human-authored, not AI-derived)
    _PERSISTENT_FIELDS = frozenset({"bookmark", "last_seen"})

    def get(self, session_id: str, cache_key: str, field: str):
        """Get a cached field. Returns None if stale or missing.

        Persistent fields (bookmark, last_seen) survive cache_key invalidation
        because they are human-authored and should not be discarded when the
        session JSONL changes.
        """
        data = self._read(session_id)
        if field in self._PERSISTENT_FIELDS:
            return data.get(field)
        if data.get("cache_key") == cache_key:
            return data.get(field)
        return None

    def set(self, session_id: str, cache_key: str, field: str, value) -> None:
        """Set a cached field, updating the cache key."""
        data = self._read(session_id)
        data["cache_key"] = cache_key
        data[field] = value
        self._write(session_id, data)

    def is_recently_seen(self, session_id: str, cooldown: int = COOLDOWN_SECONDS) -> bool:
        """Check if session was viewed recently (within cooldown seconds)."""
        data = self._read(session_id)
        return (time.time() - data.get("last_seen", 0)) < cooldown

    def touch_seen(self, session_id: str) -> None:
        """Mark session as just viewed."""
        data = self._read(session_id)
        data["last_seen"] = time.time()
        self._write(session_id, data)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def _read(self, session_id: str) -> dict:
        path = self._path(session_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, session_id: str, data: dict) -> None:
        try:
            self._path(session_id).write_text(json.dumps(data, default=str))
        except OSError:
            pass
