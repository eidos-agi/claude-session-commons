"""Tests for entity extraction module."""

from claude_session_commons.entities import extract_entities


def test_extract_file_paths():
    """Extracts absolute file paths from content."""
    content = "USER: Fix the bug in /src/auth/login.py and /tests/test_auth.py"
    entities = extract_entities(content)
    paths = [v for t, v in entities if t == "file_path"]
    assert "/src/auth/login.py" in paths
    assert "/tests/test_auth.py" in paths


def test_extract_relative_paths():
    """Extracts relative file paths."""
    content = "Read ./src/config.ts and ../utils/helpers.py"
    entities = extract_entities(content)
    paths = [v for t, v in entities if t == "file_path"]
    assert "./src/config.ts" in paths
    assert "../utils/helpers.py" in paths


def test_extract_error_classes():
    """Extracts Python-style error class names."""
    content = "Got a ValueError and then a ConnectionError when calling the API"
    entities = extract_entities(content)
    errors = [v for t, v in entities if t == "error_class"]
    assert "ValueError" in errors
    assert "ConnectionError" in errors


def test_extract_urls():
    """Extracts HTTP URLs."""
    content = "Check https://github.com/org/repo/issues/42 for details"
    entities = extract_entities(content)
    urls = [v for t, v in entities if t == "url"]
    assert any("github.com" in u for u in urls)


def test_extract_from_metadata():
    """Extracts file paths from metadata.files_touched."""
    content = "USER: edit the config"
    metadata = {"files_touched": ["/app/config.py", "/app/settings.yaml"]}
    entities = extract_entities(content, metadata)
    paths = [v for t, v in entities if t == "file_path"]
    assert "/app/config.py" in paths
    assert "/app/settings.yaml" in paths


def test_deduplication():
    """Same entity not extracted twice."""
    content = "Edit /src/main.py then read /src/main.py again"
    metadata = {"files_touched": ["/src/main.py"]}
    entities = extract_entities(content, metadata)
    paths = [v for t, v in entities if t == "file_path"]
    assert paths.count("/src/main.py") == 1


def test_skips_dev_null():
    """Skips common false positive paths."""
    content = "Redirect to /dev/null"
    entities = extract_entities(content)
    paths = [v for t, v in entities if t == "file_path"]
    assert "/dev/null" not in paths


def test_empty_content():
    """Handles empty content gracefully."""
    assert extract_entities("") == []
    assert extract_entities("", {}) == []


def test_mixed_entities():
    """Extracts multiple entity types from one chunk."""
    content = """USER: Fix the ImportError in /src/app.py
ASSISTANT: The issue is a missing import. See https://docs.python.org/3/library/importlib.html"""
    entities = extract_entities(content)
    types = set(t for t, v in entities)
    assert "file_path" in types
    assert "error_class" in types
    assert "url" in types
