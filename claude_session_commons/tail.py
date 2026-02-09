"""JSONL tail reader.

Reads the last 16KB of a session JSONL file to extract the most recent
timestamp and the type of the last entry. This avoids reading the entire
file for session discovery and sorting.
"""

import json
from datetime import datetime
from pathlib import Path

TAIL_BYTES = 16384  # 16KB — enough to get several entries from end of file


def get_tail_info(jsonl_file: Path) -> dict:
    """Read last 16KB of a JSONL for timestamp and entry type.

    Returns dict with:
        timestamp: float | None — epoch time of most recent entry
        last_entry_type: str | None — type field of last parseable entry
    """
    info = {"timestamp": None, "last_entry_type": None}
    try:
        with open(jsonl_file, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, TAIL_BYTES)
            f.seek(size - read_size)
            chunk = f.read().decode("utf-8", errors="replace")

        for line in reversed(chunk.strip().split("\n")):
            try:
                obj = json.loads(line)
                if info["last_entry_type"] is None:
                    info["last_entry_type"] = obj.get("type", "")
                ts = obj.get("timestamp")
                if ts and info["timestamp"] is None:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    info["timestamp"] = dt.timestamp()
                if info["timestamp"] is not None and info["last_entry_type"] is not None:
                    break
            except (json.JSONDecodeError, ValueError):
                continue
    except OSError:
        pass
    return info
