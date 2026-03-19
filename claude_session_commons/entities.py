"""Deterministic entity extraction from chunk content.

Extracts file paths, error classes, URLs, and project names from
session chunk text using high-precision regex patterns. No LLM needed.

Used by insights.index_session() to populate the entities table,
enabling faceted search ("show me all sessions that touched auth.py").
"""

import re
from pathlib import PurePosixPath


# Compiled patterns for performance
_FILE_PATH_RE = re.compile(
    r'(?:^|[\s(=\'"])(/[\w./_-]+\.\w{1,10})\b'
)
_RELATIVE_PATH_RE = re.compile(
    r'(?:^|[\s(=\'"])(\.{1,2}/[\w./_-]+\.\w{1,10})\b'
)
_ERROR_CLASS_RE = re.compile(
    r'\b([A-Z]\w*(?:Error|Exception|Warning|Fault))\b'
)
_URL_RE = re.compile(
    r'https?://[^\s<>"\')\]]{5,}'
)

# Skip common false-positive file paths
_SKIP_PATHS = {
    "/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr",
    "/tmp", "/etc/hosts", "/usr/bin/env",
}


def extract_entities(content: str, metadata: dict | None = None) -> list[tuple[str, str]]:
    """Extract structured entities from chunk content and metadata.

    Returns list of (entity_type, value) tuples. Types:
        - 'file_path': absolute or relative file paths
        - 'error_class': Python/JS-style error class names
        - 'url': HTTP(S) URLs

    Deduplicates within the same chunk.
    """
    metadata = metadata or {}
    seen = set()
    entities = []

    def _add(etype: str, value: str):
        key = (etype, value)
        if key not in seen:
            seen.add(key)
            entities.append(key)

    # File paths from content (absolute)
    for m in _FILE_PATH_RE.finditer(content):
        path = m.group(1)
        if path not in _SKIP_PATHS and len(path) > 3:
            _add("file_path", path)

    # File paths from content (relative ./  ../)
    for m in _RELATIVE_PATH_RE.finditer(content):
        _add("file_path", m.group(1))

    # File paths from metadata.files_touched
    for fp in metadata.get("files_touched", []):
        if isinstance(fp, str) and len(fp) > 3:
            _add("file_path", fp)

    # Error classes
    for m in _ERROR_CLASS_RE.finditer(content):
        _add("error_class", m.group(1))

    # URLs
    for m in _URL_RE.finditer(content):
        url = m.group(0).rstrip(".,;:)")
        _add("url", url)

    return entities
