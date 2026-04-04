"""Size-bucketed session cache.

One JSON file per session, keyed on cache_key = md5(path:size_bucket).
Size is bucketed so that small appends (new messages in an active session)
don't invalidate cached summaries. The cache refreshes when the session
grows meaningfully (>10% or >50KB beyond what was cached).

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

    # Minimum growth (bytes) before cache invalidates. Prevents active sessions
    # from busting cache on every new message while still refreshing when the
    # session has grown meaningfully.
    _MIN_GROWTH_BYTES = 51_200  # 50 KB
    _MIN_GROWTH_RATIO = 0.10    # 10%

    def cache_key(self, session_file: Path) -> str:
        """Create invalidation key from path + bucketed file size.

        Buckets the file size so that small appends (a few new messages)
        don't change the key. The cache refreshes when the file has grown
        by >50KB or >10% beyond the bucket boundary — whichever is larger.
        """
        try:
            size = session_file.stat().st_size
        except OSError:
            size = 0
        bucket = self._size_bucket(size)
        return hashlib.md5(f"{session_file}:{bucket}".encode()).hexdigest()

    @classmethod
    def _size_bucket(cls, size: int) -> int:
        """Round size down to a bucket boundary.

        Bucket width = max(50KB, 10% of size). This means:
        - 100KB file → bucket width 50KB → invalidates at 150KB
        - 1MB file   → bucket width 100KB → invalidates at 1.1MB
        - 10MB file  → bucket width 1MB   → invalidates at 11MB
        """
        width = max(cls._MIN_GROWTH_BYTES, int(size * cls._MIN_GROWTH_RATIO))
        return (size // width) * width

    # Fields that persist across cache_key changes.
    # - Human-authored fields (bookmark, last_seen) persist because the user wrote them.
    # - classification persists because session origin (human vs AI-spawned) is an
    #   intrinsic property of the session — it doesn't change when new messages arrive.
    _PERSISTENT_FIELDS = frozenset({"bookmark", "last_seen", "classification"})

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
