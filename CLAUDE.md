# CLAUDE.md — claude-session-commons

Shared session discovery, caching, classification, and display helpers for Claude Code tools. Used by both `claude-boss` and `claude-resume`.

## Hard Constraints

- **Never import `anthropic` SDK.** Use `claude -p` for LLM calls.
- **No files over 2700 lines.** One concern per module.
- **Conservative defaults.** When in doubt, show the session (never hide real sessions).
- **No UI code.** This package is pure data/logic — no Textual, no terminal rendering.

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `cache.py` | `SessionCache` — mtime-based JSON cache per session |
| `discovery.py` | `find_all_sessions()`, `find_recent_sessions()` — scan ~/.claude/projects/ |
| `paths.py` | `decode_project_path()` (greedy), `shorten_path()` |
| `tail.py` | `get_tail_info()` — read JSONL tail for timestamp + entry type |
| `display.py` | `relative_time()`, `format_size()`, `get_date_group()`, `_plural()` |
| `classify.py` | `quick_scan()`, `classify_session()`, ML ensemble, `get_label()` |
| `parse.py` | `parse_session()` — full JSONL parsing for summaries |
| `scoring.py` | `interruption_score()`, `has_uncommitted_changes()` |
| `git_context.py` | `get_git_context()` — git repo metadata |
| `export.py` | `export_context_md()` — markdown briefing generation |

## Testing

```bash
python -m pytest tests/ -v
```
