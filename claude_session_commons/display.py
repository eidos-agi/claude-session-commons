"""Display helpers for session metadata.

Formatting functions for time, size, and date grouping used by both
claude-boss and claude-resume TUIs.
"""

import time
from datetime import datetime, timedelta


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def relative_time(mtime: float, *, compact: bool = False) -> str:
    """Human-readable relative time.

    Default (compact=False): "3 hours, 12 minutes ago" (claude-resume style)
    Compact (compact=True): "3h ago" (claude-boss style)
    """
    delta = int(time.time() - mtime)
    if delta < 60:
        return "just now"

    minutes = delta // 60
    hours = delta // 3600
    days = delta // 86400

    if compact:
        if delta < 3600:
            return f"{minutes}m ago"
        elif delta < 86400:
            return f"{hours}h ago"
        else:
            return f"{days}d ago"

    if delta < 3600:
        return f"{_plural(minutes, 'minute')} ago"
    elif delta < 86400:
        remaining_min = (delta % 3600) // 60
        if remaining_min:
            return f"{_plural(hours, 'hour')}, {_plural(remaining_min, 'minute')} ago"
        return f"{_plural(hours, 'hour')} ago"
    else:
        remaining_hrs = (delta % 86400) // 3600
        if remaining_hrs:
            return f"{_plural(days, 'day')}, {_plural(remaining_hrs, 'hour')} ago"
        return f"{_plural(days, 'day')} ago"


def get_date_group(mtime: float) -> str:
    """Bucket a timestamp into date groups for list section headers."""
    now = datetime.now()
    dt = datetime.fromtimestamp(mtime)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=7)
    month_start = today_start - timedelta(days=30)

    if dt >= today_start:
        return "Today"
    elif dt >= yesterday_start:
        return "Yesterday"
    elif dt >= week_start:
        return "Last 7 Days"
    elif dt >= month_start:
        return "Last 30 Days"
    else:
        return "Older"


def format_size(size_bytes: int) -> str:
    """Format file size: '14.2 MB'."""
    size_bytes = max(0, size_bytes)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_duration(secs: float) -> str:
    """Human-readable duration string: '2h 15m'."""
    if secs < 60:
        return f"{int(secs)}s"
    minutes = int(secs) // 60
    if secs < 3600:
        remaining = int(secs) % 60
        return f"{minutes}m {remaining}s" if remaining else f"{minutes}m"
    hours = int(secs) // 3600
    remaining_min = (int(secs) % 3600) // 60
    return f"{hours}h {remaining_min}m" if remaining_min else f"{hours}h"
