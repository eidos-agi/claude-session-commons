# QA Instructions — claude-session-commons

Shared session discovery, caching, classification, summarization, and transcript intelligence for Claude Code tools.

## Run Tests

### Prerequisites

```bash
cd ~/repos-rheaimpact/claude-session-commons
pip install -e ".[insights]"
```

### Full Suite (excluding TUI)

```bash
python -m pytest tests/ -v --ignore=tests/test_tui.py
```

Expected: **34 tests** across 3 test files:
- `test_chunkers.py` — 16 tests (JSONL parsing, turn/subagent chunking)
- `test_insights.py` — 13 tests (sqlite-vec, fastembed, indexing, vector search)
- `test_cli_find.py` — 5 tests (CLI smoke tests)

### TUI Tests Only

```bash
python -m pytest tests/test_tui.py -v
```

Requires a display/terminal environment (may fail in headless CI).

### Chunker Tests Only (fast, no deps)

```bash
python -m pytest tests/test_chunkers.py -v
```

These run in ~0.3s and don't need sqlite-vec or fastembed.

### Insights Tests Only (needs [insights] extra)

```bash
python -m pytest tests/test_insights.py -v
```

First run downloads the fastembed model (~90MB to `~/.cache/fastembed/`). Subsequent runs are fast (~2s).

## Check Logs

### Daemon Log

The session daemon logs to `~/.claude/daemon.log` (5MB rotating, 3 backups).

```bash
tail -50 ~/.claude/daemon.log
```

Look for:
- `OK:` lines — successfully processed sessions
- `FAIL:` lines — sessions that failed to process
- `INSIGHTS:` lines — insights indexing results (turns + subagents indexed)
- `INSIGHTS FAIL:` lines — embedding/indexing errors
- `TASK OK:` — TUI-requested deep dives or patterns analysis

### Daemon Status

```bash
cat ~/.claude/session-daemon.pid && ps -p $(cat ~/.claude/session-daemon.pid) -o pid,etime,rss,command
```

Expected: daemon running, RSS < 200MB, uptime matching system uptime.

### Insights Database Stats

```bash
claude-find --stats
```

Shows: sessions indexed, total chunks, turns, subagent summaries.

## Health Check

### 1. Daemon is alive

```bash
kill -0 $(cat ~/.claude/session-daemon.pid 2>/dev/null) 2>/dev/null && echo "DAEMON: OK" || echo "DAEMON: NOT RUNNING"
```

### 2. Insights DB exists and is valid

```bash
python3 -c "
from claude_session_commons.insights import get_db, init_db, get_stats
conn = get_db()
init_db(conn)
stats = get_stats(conn)
print(f'INSIGHTS DB: OK — {stats[\"total_chunks\"]} chunks across {stats[\"sessions_indexed\"]} sessions')
conn.close()
"
```

### 3. Embedding model cached

```bash
ls ~/.cache/fastembed/models--BAAI--bge-small-en-v1.5/ > /dev/null 2>&1 && echo "EMBEDDING MODEL: OK (cached)" || echo "EMBEDDING MODEL: NOT CACHED (will download on first use)"
```

### 4. Session cache directory

```bash
ls ~/.claude/session-cache/*.json 2>/dev/null | wc -l | xargs -I{} echo "SESSION CACHE: {} cached sessions"
```

## Smoke Test

Quick end-to-end verification of the insights pipeline.

### 1. Index a session

```bash
python3 -c "
from claude_session_commons.insights import get_db, init_db, index_session, get_stats
from fastembed import TextEmbedding
from pathlib import Path

model = TextEmbedding('BAAI/bge-small-en-v1.5')
conn = get_db()
init_db(conn)

# Find a real session to index
sessions_dir = Path.home() / '.claude' / 'projects'
for project_dir in sessions_dir.iterdir():
    for jsonl in project_dir.glob('*.jsonl'):
        if jsonl.stat().st_size > 10000:
            turns, subs = index_session(str(jsonl), conn, model=model, summarize_subagents=False)
            if turns > 0:
                print(f'Indexed {jsonl.name}: {turns} turns, {subs} subagents')
                break
    else:
        continue
    break

stats = get_stats(conn)
print(f'Total: {stats[\"total_chunks\"]} chunks across {stats[\"sessions_indexed\"]} sessions')
conn.close()
"
```

### 2. Search

```bash
claude-find "what was being worked on" --limit 3
```

Expected: Results with `[distance] timestamp | chunk_type | project_path` format.

### 3. Type-filtered search

```bash
claude-find "research findings" --type subagent_summary --limit 3
```

Expected: Only `subagent_summary` chunk types in results (or "No results" if none indexed).

## Dependencies

| Package | Extra | Purpose |
|---------|-------|---------|
| numpy | core | ML feature arrays (classify.py) |
| textual>=0.40 | core | TUI framework |
| sqlite-vec | insights | Vector storage extension for SQLite |
| fastembed>=0.2.0 | insights | Local ONNX embeddings (BAAI/bge-small-en-v1.5) |
| pysqlite3 | insights | SQLite with extension loading support |
| pandas | train | ML training data prep |
| scikit-learn | train | ML classifier training |

Install all insights deps: `pip install -e ".[insights]"`
