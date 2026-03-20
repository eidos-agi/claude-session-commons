# claude-session-commons

Shared session discovery, caching, classification, and summarization for Claude Code tools.

Used by:
- **[claude-resume](https://github.com/eidos-agi/claude-resume)** — crash recovery TUI + MCP server for searching and resuming past sessions
- **[claude-boss](https://github.com/eidos-agi/claude-boss)** — Claude Code orchestration daemon
- **[claude-resume-duet](https://github.com/eidos-agi/claude-resume-duet)** — web UI companion with session browser and URL scheme handler

Also ships its own standalone tools:

| Command | What it does |
|---------|-------------|
| `claude-session-daemon` | Background daemon that pre-indexes and summarizes sessions as you work |
| `claude-find` | CLI for searching sessions from the terminal |
| `claude-session-web` | Local web UI for browsing session history |
| `claude-session-reindex` | Force a full re-index of all sessions |

## Install

```bash
git clone https://github.com/eidos-agi/claude-session-commons
cd claude-session-commons
pip install -e .
```

## Modules

| Module | Purpose |
|--------|---------|
| `cache.py` | `SessionCache` — mtime-based JSON cache per session |
| `discovery.py` | `find_all_sessions()`, `find_recent_sessions()` — scan `~/.claude/projects/` |
| `paths.py` | `decode_project_path()`, `shorten_path()` |
| `parse.py` | `parse_session()` — full JSONL parsing for summaries |
| `scoring.py` | `interruption_score()`, `has_uncommitted_changes()` — lifecycle-aware scoring |
| `classify.py` | ML ensemble to separate human sessions from automated/bot sessions |
| `summarize.py` | `summarize_quick()`, `summarize_deep()`, `analyze_patterns()` via `claude -p` |
| `export.py` | `export_context_md()` — markdown briefing with bookmark support |
| `git_context.py` | `get_git_context()` — git repo metadata |
| `display.py` | `relative_time()`, `format_size()`, `lifecycle_badge()` |
| `tail.py` | `get_tail_info()` — efficient JSONL tail reads |
| `tui/` | Embeddable Textual widgets — session picker, search input, preview pane |
| `daemon.py` | Background summarization daemon |

## Session Bookmarks

The cache stores an optional `bookmark` field per session, written by the `/bookmark` Claude Code skill or the auto-bookmark Stop hook:

- `lifecycle_state`: `done` | `paused` | `blocked` | `handing-off` | `auto-closed`
- `context`: `{summary, goal, current_step}`
- `workspace_state`: `{dirty, uncommitted_files, last_commit, diff_summary}`
- `next_actions`, `blockers`, `confidence`

`scoring.py` uses lifecycle state to override heuristic interruption scores. `display.py` renders lifecycle badges in the TUI. `export.py` includes bookmark data in markdown exports.

## Testing

```bash
python -m pytest tests/ -v
```

## License

MIT
