# CLAUDE.md — claude-session-commons

Shared session discovery, caching, classification, and display helpers for Claude Code tools. Used by both `claude-boss` and `claude-resume`.

## Hard Constraints

- **Never import `anthropic` SDK.** Use `claude -p` for LLM calls.
- **No files over 2700 lines.** One concern per module.
- **Conservative defaults.** When in doubt, show the session (never hide real sessions).
- **TUI code lives in `tui/` subpackage only.** Core data modules have no Textual dependency.

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `cache.py` | `SessionCache` — mtime-based JSON cache per session |
| `discovery.py` | `find_all_sessions()`, `find_recent_sessions()` — scan ~/.claude/projects/ |
| `paths.py` | `decode_project_path()` (greedy), `shorten_path()` |
| `tail.py` | `get_tail_info()` — read JSONL tail for timestamp + entry type |
| `display.py` | `relative_time()`, `format_size()`, `get_date_group()`, `lifecycle_badge()` |
| `classify.py` | `quick_scan()`, `classify_session()`, ML ensemble, `get_label()` |
| `parse.py` | `parse_session()` — full JSONL parsing for summaries |
| `scoring.py` | `interruption_score()`, `has_uncommitted_changes()` — lifecycle-aware scoring |
| `git_context.py` | `get_git_context()` — git repo metadata |
| `export.py` | `export_context_md()` — markdown briefing with bookmark support |
| `summarize.py` | `summarize_quick()`, `summarize_deep()`, `analyze_patterns()` via `claude -p` |
| `tui/widgets.py` | `SearchInput`, `DateHeader`, `SessionItem`, `TaskDone`, `PreviewMode` |
| `tui/ops.py` | `SessionOps` dataclass — callback bundle for TUI |
| `tui/session_picker.py` | `SessionPickerPanel` — embeddable Textual widget with full session picker |

## Session Bookmark Cache Field

The `SessionCache` stores an optional `bookmark` field per session, written by the `/bookmark` Claude Code skill or the auto-bookmark Stop hook. This is a JSON object with:

- `lifecycle_state`: `done` | `paused` | `blocked` | `handing-off` | `auto-closed`
- `context`: `{summary, goal, current_step}`
- `workspace_state`: `{dirty, uncommitted_files, last_commit, diff_summary}`
- `next_actions`: list of strings
- `blockers`: list of strings or `{description, severity, workaround}` objects
- `confidence`: `{level, risk_areas}`
- `meta`: `{created_by}` — `"manual"` (from /bookmark) or `"auto"` (from Stop hook)

Consumers:
- `scoring.py` reads `bookmark.lifecycle_state` to override heuristic interruption scores
- `display.py` `lifecycle_badge()` returns Rich-styled badge strings for the TUI
- `export.py` includes bookmark data in markdown exports
- `tui/widgets.py` `SessionItem` shows lifecycle badges in session list rows
- `tui/session_picker.py` renders bookmark details (next actions, blockers, confidence) in the preview pane

## Documentation

- [JSONL Schema Reference](docs/jsonl-schema.md) — Complete spec for Claude Code session transcript files (message types, fields, conversation tree structure, volume profiles, query examples)

## Testing

```bash
python -m pytest tests/ -v
```
