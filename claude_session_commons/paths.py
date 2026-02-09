"""Path decoding and display utilities.

Handles the lossy encoding Claude Code uses for project directories:
    /Users/you/repos/my-app → -Users-you-repos-my-app

The greedy decoder checks filesystem existence to recover hyphenated
directory names that would otherwise be split into separate segments.
"""

from pathlib import Path


def decode_project_path(encoded: str) -> str:
    """Decode a project directory name back to a path.

    Claude encodes /Users/you/repos/myapp as -Users-you-repos-myapp.
    This is lossy for hyphenated paths (my-app -> my/app), so we
    try the naive decode first and fall back to progressively
    keeping hyphens until we find a path that exists.
    """
    if not encoded.startswith("-"):
        return encoded.replace("-", "/")

    # Naive decode: replace all hyphens with slashes
    naive = "/" + encoded[1:].replace("-", "/")
    if Path(naive).exists():
        return naive

    # Try to find the real path by checking each segment
    parts = encoded[1:].split("-")
    best = _greedy_path_decode(parts)
    if best:
        return best

    # Fallback to naive decode even if path doesn't exist
    return naive


def _greedy_path_decode(parts: list[str]) -> str | None:
    """Greedily decode path segments, preferring longer real directory names."""
    if not parts:
        return None

    path = "/"
    i = 0
    while i < len(parts):
        # Try joining progressively more parts with hyphens
        best_end = i
        for j in range(len(parts) - 1, i - 1, -1):
            candidate = "-".join(parts[i:j + 1])
            test_path = path.rstrip("/") + "/" + candidate
            if Path(test_path).exists():
                best_end = j
                break

        segment = "-".join(parts[i:best_end + 1])
        path = path.rstrip("/") + "/" + segment
        i = best_end + 1

    return path


def shorten_path(path: str) -> str:
    """Replace home directory with ~. Returns dash for empty paths."""
    if not path:
        return "\u2014"
    home = str(Path.home())
    if path.startswith(home):
        return "~" + path[len(home):]
    return path
