# Claude Code Session JSONL Schema

Claude Code stores each session as a `.jsonl` file in `~/.claude/projects/<encoded-path>/`. Each line is a JSON object with a `type` field that determines its structure.

## Message Types

### `user` — Your messages

| Field | Description |
|-------|-------------|
| `message.content` | What you typed (string or content array) |
| `cwd` | Working directory at the time |
| `sessionId` | Session UUID |
| `uuid` | This message's unique ID |
| `parentUuid` | Links to the message it's replying to (conversation tree) |
| `timestamp` | ISO 8601 timestamp |
| `permissionMode` | Active permission mode (e.g., `plan`, `default`) |
| `version` | Claude Code version string |
| `gitBranch` | Current git branch |

### `assistant` — Claude's responses

| Field | Description |
|-------|-------------|
| `message.content[]` | Array of `text` and `tool_use` blocks |
| `message.model` | Which model produced the response (e.g., `claude-opus-4-6`) |
| `message.usage` | Token counts: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` |
| `parentUuid` | Links back to the user message it's responding to |
| `requestId` | Anthropic API request ID |
| `uuid` | This message's unique ID |

**Content block types within `message.content[]`:**

- `{"type": "text", "text": "..."}` — Claude's written response
- `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}` — Tool invocation (Read, Write, Edit, Bash, Glob, Grep, Task, etc.)
- `{"type": "tool_result", "tool_use_id": "...", "content": "..."}` — Result returned from tool execution

### `progress` — Subagent activity (the Task tool)

| Field | Description |
|-------|-------------|
| `data.message` | The prompt sent to the subagent |
| `slug` | Agent identifier (e.g., `resilient-roaming-frost`) |
| `type` | Always `progress` |

These are the bulk of any session that uses parallel agents. In a typical research session, `progress` entries outnumber all other types combined (e.g., 1,700+ of 4,300 total lines). The vendor research deep-dives, web searches, and code exploration all live here.

### `queue-operation` — Agent lifecycle events

| Field | Description |
|-------|-------------|
| `operation` | `enqueue` or `dequeue` |
| `content` | Task ID and description |

Tracks when subagents are spawned and when they complete. Pair with `progress` entries sharing the same `slug` to reconstruct a subagent's full execution.

### `system` — Hook outputs and system events

| Field | Description |
|-------|-------------|
| `subtype` | Event type (e.g., `stop_hook_summary`) |
| `hookInfos` | Which hooks fired and their output |
| `message` | System message content |

Includes bookmark hooks, plugin hooks, permission prompts, and context compression notifications.

### `file-history-snapshot` — File state checkpoints

| Field | Description |
|-------|-------------|
| `snapshot.trackedFileBackups` | Files being tracked for undo |
| `messageId` | Links to the message that triggered the edit |
| `isSnapshotUpdate` | Whether this updates a prior snapshot |

Bookend around file edits. Used by Claude Code's undo system to restore files to previous states.

## Conversation Tree

The `parentUuid` -> `uuid` chain forms a conversation tree. To reconstruct the dialogue:

1. **Full flow:** Follow `parentUuid` links from any message to trace the conversation path
2. **Clean back-and-forth:** Filter to `type: user` + `type: assistant` entries only
3. **Agent delegation:** Filter to `type: progress` to see subagent work (where research deep-dives are buried)
4. **Chronological:** Sort by line number (JSONL is append-only, so line order = time order)

## Volume Profile

A typical 2-3 hour interactive session produces:

| Type | Typical Count | % of File |
|------|--------------|-----------|
| `progress` | 1,500-3,800 | ~50-55% |
| `assistant` | 1,400-2,200 | ~30-35% |
| `user` | 900-1,400 | ~15-20% |
| `file-history-snapshot` | 40-210 | ~2-5% |
| `queue-operation` | 25-280 | ~1-3% |
| `system` | 85-170 | ~1-2% |

File sizes range from 50KB (short sessions) to 25MB+ (extended sessions with heavy agent delegation).

## Useful Queries

**Extract just the human conversation:**
```python
[line for line in jsonl if line["type"] in ("user", "assistant")]
```

**Find all tool invocations:**
```python
[block for line in jsonl if line["type"] == "assistant"
 for block in line["message"]["content"] if block["type"] == "tool_use"]
```

**Reconstruct a subagent's work:**
```python
[line for line in jsonl if line["type"] == "progress" and line.get("slug") == "agent-slug"]
```

**Get token usage per response:**
```python
[(line["message"]["model"], line["message"]["usage"])
 for line in jsonl if line["type"] == "assistant" and "usage" in line.get("message", {})]
```

**Find all files that were edited:**
```python
set(block["input"]["file_path"]
    for line in jsonl if line["type"] == "assistant"
    for block in line.get("message", {}).get("content", [])
    if block.get("type") == "tool_use" and block.get("name") in ("Edit", "Write"))
```
