# SPEC: Session Transcript Intelligence

**Implements:** [PRD.md](PRD.md)
**Author:** Daniel Shanklin + Claude Opus 4.6
**Date:** 2026-02-19
**Status:** Draft

---

## 1. Data Model

### 1.1 Database Location

```
~/.claude/insights.db
```

Single SQLite file. Created on first run by `init_db()`.

### 1.2 Schema

```sql
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,            -- UUID as text
    session_id TEXT NOT NULL,       -- Original sessionId from JSONL
    project_path TEXT NOT NULL,     -- Decoded project path
    chunk_type TEXT NOT NULL        -- 'turn' | 'subagent_summary'
        CHECK(chunk_type IN ('turn', 'subagent_summary')),
    content TEXT NOT NULL,          -- Embedded text content
    metadata TEXT NOT NULL,         -- JSON as text (see 1.3)
    source_uuid TEXT,              -- UUID of originating user/assistant message
    timestamp TEXT NOT NULL,        -- ISO 8601 from source message
    indexed_at TEXT NOT NULL        -- When this chunk was created
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_ts ON chunks(timestamp);

-- sqlite-vec virtual table for vector index
-- Linked to chunks table by rowid
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    embedding float[384]           -- BAAI/bge-small-en-v1.5 dimension
);
```

**Relationship:** `chunks.rowid` = `vec_chunks.rowid`. Inserts must happen in the same transaction.

### 1.3 Metadata Schema (JSON in `metadata` column)

**For `chunk_type = 'turn'`:**
```json
{
  "user_uuid": "uuid-of-user-message",
  "assistant_uuid": "uuid-of-assistant-message",
  "model": "claude-opus-4-6",
  "tools_used": ["Edit", "Bash", "Grep"],
  "files_touched": ["/path/to/file.py"],
  "token_count": 4521
}
```

**For `chunk_type = 'subagent_summary'`:**
```json
{
  "slug": "resilient-roaming-frost",
  "initial_prompt_preview": "Research the Navusoft API...",
  "tools_used": ["WebSearch", "WebFetch", "Read"],
  "progress_line_count": 287
}
```

## 2. Chunking Pipeline

### 2.1 Turn Chunker

**Input:** Path to `.jsonl` session file
**Output:** List of `TurnChunk` objects

```python
@dataclasses.dataclass
class TurnChunk:
    user_uuid: str
    assistant_uuid: str
    content: str        # The text to embed
    metadata: dict
    timestamp: str      # From user message
```

**Algorithm:**
1. Read JSONL line by line. Build lookup: `uuid -> entry`.
2. For each `type: "user"` entry:
   a. Find the `type: "assistant"` entry where `assistant.parentUuid == user.uuid`.
   b. Extract user text from `message.content` (string or first text block in array).
   c. Extract assistant text blocks from `message.content[]` where `type == "text"`.
   d. Extract tool calls: for each `tool_use` block, capture `name` and `input` keys (not values of large inputs like file contents — just the key structure).
   e. Concatenate into `content`:
      ```
      USER: <user text>

      ASSISTANT: <assistant text>

      TOOLS: Edit(file_path=/path/to/file.py), Bash(command=git status)
      ```
   f. Truncate `content` to 2000 characters. If truncated, append `[...truncated]`.
3. Skip turns where assistant response is empty or contains only tool results.

### 2.2 Subagent Chunker

**Input:** Path to `.jsonl` session file
**Output:** List of `SubagentChunk` objects

```python
@dataclasses.dataclass
class SubagentChunk:
    slug: str
    content: str        # LLM summary of the subagent's work
    metadata: dict
    timestamp: str      # From first progress entry
```

**Algorithm:**
1. Group all `type: "progress"` entries by `slug`.
2. For each slug group with > 5 entries (skip trivial agents):
   a. Extract the initial prompt from the first entry's `data.message`.
   b. Collect all text content from subsequent entries (assistant responses within the agent).
   c. Concatenate into a summary prompt (truncated to fit context).
   d. Call `claude -p --model haiku` with prompt:
      ```
      Summarize this AI agent's work in 2-3 sentences. Focus on:
      what was researched/built, key findings, and outcome.

      Initial task: <initial prompt>

      Work performed:
      <concatenated progress content, truncated to 4000 chars>
      ```
   e. The LLM response becomes `content`.
3. Rate limit: max 3 summarizations per second (daemon constraint).

### 2.3 Session Deduplication

Before indexing a session:
1. Check if `session_id` already exists in `chunks` table.
2. If yes, check the session file's mtime against `indexed_at` of existing chunks.
3. If file is newer, delete old chunks (both `chunks` and `vec_chunks` rows) and re-index.
4. If file is same age or older, skip.

## 3. Embedding

### 3.1 Model

```python
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
# First call downloads ~90MB model to ~/.cache/fastembed/
# Subsequent calls load from cache
# Produces 384-dimensional float32 vectors
```

### 3.2 Embedding Flow

```python
def embed_chunks(chunks: list[TurnChunk | SubagentChunk], model: TextEmbedding) -> list[list[float]]:
    texts = [c.content for c in chunks]
    embeddings = list(model.embed(texts))  # Batch encode
    return embeddings
```

### 3.3 Storage

```python
import sqlite_vec
import struct

def serialize_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)

# In transaction:
cursor.execute("INSERT INTO chunks (...) VALUES (...)")
chunk_rowid = cursor.lastrowid
cursor.execute(
    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
    (chunk_rowid, serialize_vector(embedding))
)
```

## 4. Query API

### 4.1 Module: `claude_session_commons/insights.py`

```python
import dataclasses
import json
import sqlite3
from typing import List, Optional
from fastembed import TextEmbedding

DB_PATH = "~/.claude/insights.db"

@dataclasses.dataclass
class ChunkResult:
    id: str
    session_id: str
    project_path: str
    chunk_type: str
    timestamp: str
    content: str
    metadata: dict
    distance: float

def get_db(path: str = DB_PATH) -> sqlite3.Connection:
    """Open insights DB with sqlite-vec extension loaded."""
    ...

def init_db(conn: sqlite3.Connection):
    """Create tables and indexes if they don't exist."""
    ...

def index_session(
    session_path: str,
    conn: sqlite3.Connection,
    model: TextEmbedding
) -> tuple[int, int]:
    """
    Parse, chunk, embed, and store a session.
    Returns (turns_indexed, subagents_indexed).
    Called by the daemon.
    """
    ...

def query(
    text: str,
    conn: sqlite3.Connection,
    model: TextEmbedding,
    limit: int = 10
) -> List[ChunkResult]:
    """
    Semantic search over all indexed chunks.
    Returns results sorted by distance (ascending = most similar).
    Called by claude-find and other consumers.
    """
    ...
```

### 4.2 Query Implementation

```python
def query(text, conn, model, limit=10):
    # 1. Embed the query
    query_vec = list(model.embed([text]))[0]
    query_bytes = serialize_vector(query_vec)

    # 2. Vector search via sqlite-vec
    rows = conn.execute("""
        SELECT v.rowid, v.distance
        FROM vec_chunks v
        WHERE v.embedding MATCH ?
        ORDER BY v.distance
        LIMIT ?
    """, (query_bytes, limit)).fetchall()

    # 3. Fetch chunk metadata
    rowids = [r[0] for r in rows]
    distances = {r[0]: r[1] for r in rows}

    placeholders = ",".join("?" * len(rowids))
    chunks = conn.execute(f"""
        SELECT rowid, id, session_id, project_path, chunk_type,
               timestamp, content, metadata
        FROM chunks
        WHERE rowid IN ({placeholders})
    """, rowids).fetchall()

    # 4. Assemble results
    results = []
    for row in chunks:
        results.append(ChunkResult(
            id=row[1],
            session_id=row[2],
            project_path=row[3],
            chunk_type=row[4],
            timestamp=row[5],
            content=row[6],
            metadata=json.loads(row[7]),
            distance=distances[row[0]]
        ))

    return sorted(results, key=lambda r: r.distance)
```

## 5. CLI: `claude-find`

### 5.1 Entry Point

Add to `pyproject.toml`:
```toml
[project.scripts]
claude-session-daemon = "claude_session_commons.daemon:main"
claude-find = "claude_session_commons.cli_find:main"
```

### 5.2 Usage

```bash
claude-find "what mistakes were made with Navusoft?"
claude-find "how did we handle WAM integration?" --limit 5
claude-find "user corrections about email format" --type turn
```

### 5.3 Output Format

```
[0.23] 2026-02-18 14:32  |  turn  |  ~/repos-greenmark-waste-solutions
  USER: it's not all six, there are many more, do more research, fact find.
  ASSISTANT: You're right — I was only listing 6 vendors but there are 15 total
  in the vendor-status.md. Let me read all 9 remaining vendor docs...
  TOOLS: Read(infra/vendors/3rd-eye.md), Read(infra/vendors/lb-technologies.md)

[0.31] 2026-02-18 13:15  |  subagent_summary  |  ~/repos-greenmark-waste-solutions
  Researched WAM Software Inc (Reno, NV) as the WAM vendor. Confirmed NO API
  exists. Proposed 12 bronze tables inferred from industry patterns. Recommended
  CSV export for short-term integration, direct DB access for long-term.

[0.34] 2026-02-18 12:40  |  turn  |  ~/repos-greenmark-waste-solutions
  USER: do you yet have a process to document all of this to yourself as you go?
  ASSISTANT: Honest answer: devlogs and MEMORY.md exist but I don't commit as
  I go, only when asked...
```

Format: `[distance] timestamp | chunk_type | project_path` followed by truncated content.

## 6. Daemon Integration

### 6.1 Hook Point

In `daemon.py`, after existing session summarization, add:

```python
from claude_session_commons.insights import get_db, init_db, index_session

# During daemon startup:
insights_conn = get_db()
init_db(insights_conn)
insights_model = TextEmbedding("BAAI/bge-small-en-v1.5")

# In the session processing loop, after summarization:
turns, subagents = index_session(session_path, insights_conn, insights_model)
```

### 6.2 Model Loading

The fastembed model is loaded once at daemon startup and reused. First-ever run downloads ~90MB to `~/.cache/fastembed/`. Subsequent runs load from disk in ~2 seconds.

## 7. Dependencies

### New (add to pyproject.toml)

```toml
[project.optional-dependencies]
insights = ["sqlite-vec", "fastembed>=0.2.0"]
train = ["pandas", "scikit-learn"]
```

The `insights` extra keeps the heavy embedding dependencies optional. Core session-commons functionality doesn't require them.

### Install for full stack

```bash
pip install -e ".[insights]"
```

## 8. File Map

```
claude_session_commons/
    insights.py         # NEW — query(), index_session(), init_db()
    chunkers.py         # NEW — TurnChunker, SubagentChunker
    cli_find.py         # NEW — claude-find entry point
    daemon.py           # MODIFIED — add insights indexing after summarization
    summarize.py        # EXISTING — reused for subagent summaries
    discovery.py        # EXISTING — reused for session enumeration
    parse.py            # EXISTING — may extend for structured parsing
docs/
    jsonl-schema.md     # EXISTING — reference for chunking logic
planning/
    PRD.md              # NEW — this feature's requirements
    SPEC.md             # NEW — this document
```

## 9. Testing Strategy

### Unit Tests

- `test_chunkers.py` — Parse a known JSONL fixture, verify correct turn/subagent grouping
- `test_insights.py` — Index a small session, query it, verify results come back
- `test_cli_find.py` — Smoke test the CLI entry point

### Integration Test

- Index 3 real sessions from `~/.claude/projects/`
- Run 5 target queries from the PRD
- Verify each returns at least 1 relevant result in top 5

### Fixture

Create `tests/fixtures/sample_session.jsonl` with ~50 lines covering all message types, at least 2 user-assistant turns, and 1 subagent with 10+ progress entries.

## 10. Implementation Order

1. `chunkers.py` — Turn and subagent parsing (no embedding yet)
2. `insights.py` — `init_db()`, `index_session()`, `query()` with sqlite-vec
3. `cli_find.py` — Wire up the CLI
4. Tests — Unit + integration
5. `daemon.py` — Hook in indexing after summarization
6. Commit, test end-to-end with real sessions
