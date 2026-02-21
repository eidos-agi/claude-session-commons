# Planning Instructions — claude-session-commons

> Shared session discovery, caching, classification, summarization, and transcript intelligence for Claude Code tools. Used by claude-boss, claude-resume, and claude-find.

## Data Sources

- **Repo**: rheaimpact/claude-session-commons (private)
- **Local path**: `~/repos-rheaimpact/claude-session-commons`
- **Issues**: None open (GitHub Issues not yet used)
- **Devlogs**: `claude-session-commons` service in taskr (start logging here)
- **Related repos**:
  - `~/repos-rheaimpact/claude-boss` — session picker consumer
  - `~/repos-rheaimpact/claude-resume` — session resume consumer
- **Planning docs**: `planning/PRD.md`, `planning/SPEC.md` (Session Transcript Intelligence)
- **QA docs**: `qa/README.md` (34 tests, health checks, smoke tests)

## Status Review

### Gather
1. `git log --oneline -10` in the repo
2. `git status --short` for uncommitted work
3. `python -m pytest tests/ -v --ignore=tests/test_tui.py` — test suite status
4. Read `qa/README.md` health check commands
5. Read `planning/PRD.md` and `planning/SPEC.md` for feature scope
6. Check daemon status: `cat ~/.claude/session-daemon.pid && ps -p $(cat ~/.claude/session-daemon.pid)`
7. Check insights DB: `claude-find --stats`
8. Search devlogs: `devlog_search(query="session-commons", service_name="claude-session-commons")`

### Synthesize
Produce a status snapshot:
- **State**: active / stalled / blocked / shipping
- **Last commit**: date + message
- **Uncommitted work**: count + summary (new files, modified files)
- **Test suite**: pass/fail counts
- **Daemon**: running / stopped, PID, RSS
- **Insights DB**: chunks indexed, sessions indexed
- **Momentum**: high / medium / low

### Output
Present to user inline. Save to `planning/status-<YYYY-MM-DD>.md` if requested.

## Sprint Plan

### Gather
1. Read `planning/PRD.md` MVP scope (In vs Out sections)
2. Read `planning/SPEC.md` implementation order
3. `git status --short` for in-progress work
4. Read `qa/README.md` for test status
5. Check the `/ideas` output (if available) for prioritized next features
6. Search devlogs for recent decisions: `devlog_search(query="session-commons decision")`

### Synthesize
Split work into:

**Unblocked (can do now):**
- Infrastructure improvements (daemon health, embedding versioning)
- Code cleanup (remove claude -p from classification, simplify subagent summarization)
- New features that don't need external input

**Blocked (needs something):**
- Features requiring consumer changes (claude-boss, claude-resume TUI integration)
- Features requiring real-world usage data

Prioritize by: (1) infrastructure fixes, (2) simplification/removal, (3) new features.

Include estimated effort: Tiny (<2h), Small (half day), Medium (1-2 days).

### Output
Present to user inline.

## Roadmap Update

### Gather
1. Read `planning/PRD.md` — full MVP scope and deferred items
2. Read `planning/SPEC.md` — implementation order
3. `git log --all --oneline` — what's been built
4. Read `qa/README.md` — current test count vs expected
5. Check `/ideas` output for product direction (daemon health, embedding versioning, token accounting, delta resume pack)

### Synthesize
- **MVP status**: which SPEC sections are complete vs remaining
- **Post-MVP queue**: ranked from `/ideas` portfolio (Quick Wins first, then Compounding Systems)
- **Technical debt**: items from "Stop Doing" list (claude -p for classification, subagent re-parsing, etc.)
- **Consumer integration**: when to wire insights into claude-resume TUI and claude-boss

### Output
Present to user inline. Update `planning/ROADMAP.md` if it exists (create if approved).

## Project Health

### Gather
1. `git log --since="7 days ago" --oneline` — recent commits
2. `python -m pytest tests/ -v --ignore=tests/test_tui.py` — test results
3. Daemon health: PID check, RSS, `~/.claude/daemon.log` tail
4. Insights DB: `claude-find --stats`
5. Check if fastembed model is cached: `ls ~/.cache/fastembed/models--BAAI--bge-small-en-v1.5/`

### Synthesize
Score each dimension (green/amber/red):

| Dimension | Green | Amber | Red |
|-----------|-------|-------|-----|
| Tests | All 34 pass | 1-3 failures | >3 failures or can't run |
| Daemon | Running, RSS <200MB | Running but RSS >200MB | Not running |
| Insights DB | >0 chunks indexed | DB exists but empty | DB missing |
| Embedding model | Cached in ~/.cache/fastembed | Not cached (will download) | fastembed not installed |
| Uncommitted work | Clean or <5 files | 5-15 uncommitted files | >15 uncommitted files |
| Dependencies | `pip install -e ".[insights]"` works | Warnings during install | Install fails |

### Output
Present as a quick scorecard.

## Ideas Portfolio

Reference from the `/ideas` skill run (2026-02-20). Four AI models consulted.

### Quick Wins (do this week)
1. **Daemon Health & Processing Status** — Write `daemon.status.json` heartbeat (Tiny, <2h)
2. **Embedding Model Versioning** — Add `insights_meta` table tracking model/version per chunk (Tiny, ~2h)

### Compounding Systems (small-medium effort)
3. **Session Cost & Token Accounting** — Aggregate token usage from JSONL (Small, half day)
4. **Delta Resume Pack + Branch Grouping** — Show "what changed" on resume, group by git branch (Small-Medium, 1-2 days)

### Stop Doing (confirmed by multiple advisors)
- Stop calling `claude -p` for classification — use heuristics
- Stop using `claude -p` for subagent summarization — use first/last progress entry
- Stop re-parsing JSONL on every TUI open — cache aggressively
- Stop embedding subagent chunks separately — low retrieval value

### If You Only Build One Thing
Daemon Health & Processing Status (`daemon.status.json`).

## Module Map

| Module | Lines | Purpose | Status |
|--------|-------|---------|--------|
| `cache.py` | ~100 | Mtime-based JSON cache | Stable |
| `discovery.py` | ~80 | Session file scanner | Stable |
| `paths.py` | ~40 | Path encoding/decoding | Stable |
| `tail.py` | ~50 | JSONL tail reader | Stable |
| `display.py` | ~60 | Time/size formatting | Stable |
| `classify.py` | ~200 | ML session classification | Stable (needs heuristic simplification) |
| `parse.py` | ~150 | Full JSONL parser | Stable |
| `scoring.py` | ~80 | Interruption scoring | Stable |
| `git_context.py` | ~60 | Git repo metadata | Stable |
| `export.py` | ~100 | Markdown export | Stable |
| `summarize.py` | ~200 | Quick/deep summaries via claude -p | Stable |
| `chunkers.py` | ~265 | Turn + subagent JSONL chunking | New (uncommitted) |
| `insights.py` | ~240 | sqlite-vec + fastembed indexing/query | New (uncommitted) |
| `cli_find.py` | ~100 | claude-find CLI | New (uncommitted) |
| `daemon.py` | ~385 | Background processing daemon | Modified (uncommitted) |
| `tui/` | ~500 | Textual TUI widgets | Stable |
