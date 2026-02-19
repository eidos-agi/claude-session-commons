# PRD: Session Transcript Intelligence

**Product:** claude-session-commons / insights module
**Author:** Daniel Shanklin + Claude Opus 4.6
**Date:** 2026-02-19
**Status:** Draft
**Consulted:** Gemini 2.5 Pro (4 rounds, socratic design)

---

## Problem

Claude Code sessions generate rich transcripts (JSONL files) that capture not just what was built, but *why* — the reasoning, corrections, dead ends, architectural decisions, and lessons learned. Today these transcripts are:

- Used only for crash recovery (claude-resume)
- Summarized at the session level (lossy)
- Not searchable by content or meaning
- Not queryable across sessions

This means institutional knowledge is locked in flat files and lost when context windows reset.

## Opportunity

500+ sessions are already indexed by the daemon. Each session contains 50KB-25MB of structured conversational data. The knowledge trapped in these transcripts includes:

- **Architectural decisions** and the reasoning behind them
- **User corrections** that reveal what the AI got wrong
- **Implementation patterns** that worked and should be reused
- **Dead ends** that shouldn't be revisited
- **Open threads** — things left unresolved

In the age of AI, not mining this data is wasteful.

## Users

| User | Need |
|------|------|
| **Daniel (primary)** | Query past sessions for decisions, patterns, and context before starting new work |
| **Claude Code agents** | Retrieve relevant past context to avoid repeating mistakes (future: RAG) |
| **claude-resume** | Enhanced session search beyond title-level summaries |
| **claude-boss** | Ground decisions in past precedent |

## Target Queries

These are the actual queries the system must answer well:

1. "What decisions were made about X?" — architectural choices, with reasoning
2. "What went wrong with Y?" — mistakes, corrections, pivots
3. "How did we implement Z?" — patterns to reuse, with tool call context
4. "What did the user correct me on?" — self-improvement for agents
5. "What's still unresolved?" — open threads across sessions

## Solution: Local Semantic Search over Transcripts

### Core Idea

Chunk session transcripts into meaningful semantic units, embed them locally, store in a local SQLite database with vector search, and expose a query API.

### Chunking Strategy

Two chunk types, from day one:

| Chunk Type | Source | What Gets Embedded |
|-----------|--------|-------------------|
| `turn` | `user` + `assistant` messages linked by `parentUuid` | User question + Claude response text + tool names/inputs (truncated — no full file contents) |
| `subagent_summary` | All `progress` lines grouped by `slug` | LLM-generated summary of subagent work (via existing `summarize.py` / `claude -p`) |

**Why both?** In practice, the most valuable content is in subagent work — parallel research, deep-dives, vendor analysis. The user-assistant back-and-forth is mostly coordination. Both are needed for complete coverage.

### Architecture

```
Claude Code (writes .jsonl)
    |
~/.claude/projects/<path>/*.jsonl
    |
claude-session-daemon (launchd, polls every 5 min)
    | NEW: parse -> chunk -> embed -> store
    |
~/.claude/insights.db (SQLite + sqlite-vec)
    |
claude-find CLI  /  insights.query()
    |
Results: content, session, timestamp, distance
```

### Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Vector DB | `sqlite-vec` | Pure C, pip-installable, macOS ARM native, no Faiss dependency |
| Embedding | `fastembed` (ONNX) | ~100MB (not 2GB torch), instant startup, BAAI/bge-small-en-v1.5 default |
| Storage | `~/.claude/insights.db` | Single file, lives alongside existing session cache |
| Indexing | Existing daemon | Already polls every 5 min, add chunking + embedding step |
| Query CLI | `claude-find` | Thin wrapper over `insights.query()` |

### Key Design Decisions

1. **Fully local** — no remote database, no API keys for embeddings, works offline
2. **Library-first** — `insights.py` module in session-commons, not a standalone service
3. **Daemon does writes** — indexing happens in the background, CLI is read-only
4. **sqlite-vec over sqlite-vss** — vss is deprecated by its author, vec is the successor
5. **fastembed over sentence-transformers** — avoids 2GB torch dependency
6. **Hybrid chunking from day one** — turns + subagent summaries, not just turns

## MVP Scope

### In

- JSONL chunk parser (turns + subagent summaries)
- fastembed integration for local embedding (384-dim vectors)
- sqlite-vec schema and database initialization
- `index_session()` function for the daemon
- `query()` function for consumers
- `claude-find` CLI tool for human use

### Out (deferred)

- Two-stage retrieval (session-level then chunk-level)
- Metadata filters in query API (tool name, date range, project)
- RAG context injection (`get_context_for_topic()`)
- TUI integration in claude-resume
- Advanced daemon robustness (re-indexing, error recovery, schema migrations)
- Live embedding during active sessions (vs. post-session batch)

## Success Criteria

1. `claude-find "what mistakes were made with Navusoft"` returns relevant chunks from the actual Greenmark research sessions
2. Query latency < 500ms for 500 sessions worth of chunks
3. No torch dependency — total install < 200MB
4. Works fully offline after initial model download
5. Daemon indexes new sessions without manual intervention

## Dependencies

- `sqlite-vec` (pip package)
- `fastembed>=0.2.0` (pip package, pulls ONNX runtime)
- Existing: `claude-session-commons`, `claude -p` (for subagent summarization)

## Risks

| Risk | Mitigation |
|------|------------|
| sqlite-vec is young (pre-1.0) | It's pure C with simple API; fallback to sqlite-vss if needed |
| fastembed model quality insufficient | Can swap to any ONNX embedding model; bge-small-en-v1.5 is well-benchmarked |
| Subagent summarization is slow/expensive | Rate-limit in daemon; summarize only on first index, cache result |
| Database grows large with many sessions | 384-dim vectors are compact; 10K chunks ~ 15MB. Monitor and add cleanup |
| Chunking misses important context | Start with two strategies, measure recall, add more chunk types if needed |

## Open Questions

1. Should `claude-find` output be formatted for human reading or structured for piping into other tools?
2. What's the right chunk size limit for turns? Large assistant responses may need truncation.
3. Should we deduplicate chunks across sessions that share context (continuation sessions)?
4. How do we handle the `file-history-snapshot` type — ignore, or extract edit metadata?
