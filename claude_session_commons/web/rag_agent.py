"""Agentic RAG — Claude Code SDK agent with insights DB tools.

Defines MCP tools that wrap the existing retrieval functions
(semantic search, entity search, session listing, etc.) and
a run_agent() async function that executes a single agent turn,
pushing SSE events to the chat state manager.
"""

import asyncio
import html
import json
import traceback
from pathlib import Path

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)
from claude_code_sdk.types import StreamEvent

from .chat_state import ChatStateManager

# ── SDK parse-message patch ────────────────────────────────
# The SDK (v0.0.25) raises MessageParseError on unrecognised CLI
# message types like "rate_limit_event", which kills the async
# generator mid-turn.  Monkeypatch the parser so unknown types
# return None instead of raising.
import claude_code_sdk._internal.client as _sdk_client  # noqa: E402

_orig_parse_message = _sdk_client.parse_message


def _lenient_parse_message(data):
    try:
        return _orig_parse_message(data)
    except Exception:
        return None

_sdk_client.parse_message = _lenient_parse_message

# ── Tool definitions ─────────────────────────────────────────

DB_PATH = str(Path.home() / ".claude" / "insights.db")


_rag_db_initialized = False

def _get_conn():
    """Open a fresh insights DB connection. Returns None if unavailable."""
    global _rag_db_initialized
    try:
        from ..insights import get_db, init_db
        if not Path(DB_PATH).exists():
            return None
        conn = get_db(DB_PATH)
        if not _rag_db_initialized:
            try:
                init_db(conn)
            except Exception:
                pass  # Schema already initialized by daemon
            _rag_db_initialized = True
        return conn
    except Exception:
        return None


def _get_model():
    """Get the shared embedding model."""
    try:
        from fastembed import TextEmbedding
        return TextEmbedding("BAAI/bge-small-en-v1.5")
    except Exception:
        return None


_embed_model_cache = None


def _get_cached_model():
    global _embed_model_cache
    if _embed_model_cache is None:
        _embed_model_cache = _get_model()
    return _embed_model_cache


@tool("session_search", "Search session transcripts by meaning. Returns the most semantically similar conversation turns.", {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Natural language search query"},
        "limit": {"type": "integer", "description": "Max results (default 8)"},
    },
    "required": ["query"],
})
async def session_search(args):
    from ..insights import query as insight_query
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: insights database not available"}]}
    try:
        model = _get_cached_model()
        results = insight_query(
            args["query"], conn, model=model,
            limit=args.get("limit", 8),
        )
        if not results:
            return {"content": [{"type": "text", "text": "No results found."}]}

        lines = [f"Found {len(results)} results:\n"]
        for i, r in enumerate(results, 1):
            snippet = r.content[:300].replace("\n", " ")
            lines.append(
                f"[{i}] distance={r.distance:.3f} | {r.chunk_type} | "
                f"{r.timestamp[:16]} | {r.project_path}\n"
                f"    Session: {r.session_id}\n"
                f"    {snippet}\n"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    finally:
        conn.close()


@tool("entity_search", "Find sessions by specific entities: file paths, error classes, URLs, git branches, or git commits.", {
    "type": "object",
    "properties": {
        "entity_type": {
            "type": "string",
            "enum": ["file_path", "error_class", "url", "git_branch", "git_commit"],
            "description": "Type of entity to search for",
        },
        "value": {"type": "string", "description": "Value to search (supports partial match)"},
        "limit": {"type": "integer", "description": "Max results (default 10)"},
    },
    "required": ["entity_type", "value"],
})
async def entity_search(args):
    from ..insights import query_by_entity
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: insights database not available"}]}
    try:
        results = query_by_entity(
            conn, args["entity_type"], args["value"],
            limit=args.get("limit", 10),
        )
        if not results:
            return {"content": [{"type": "text", "text": f"No results for {args['entity_type']}={args['value']}"}]}

        lines = [f"Found {len(results)} matches for {args['entity_type']}='{args['value']}':\n"]
        for i, r in enumerate(results, 1):
            snippet = r.content[:200].replace("\n", " ")
            lines.append(
                f"[{i}] {r.chunk_type} | {r.timestamp[:16]} | {r.project_path}\n"
                f"    Session: {r.session_id}\n"
                f"    {snippet}\n"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    finally:
        conn.close()


@tool("list_sessions", "List recently active sessions with titles and summaries.", {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "description": "Max sessions (default 15)"},
        "hours": {"type": "number", "description": "Lookback window in hours (default 720 = 30 days)"},
    },
    "required": [],
})
async def list_sessions(args):
    try:
        from ..discovery import find_recent_sessions
        from ..cache import SessionCache
        from ..display import relative_time

        sessions = find_recent_sessions(
            args.get("hours", 720),
            max_sessions=args.get("limit", 15),
        )
        cache = SessionCache()

        lines = [f"Found {len(sessions)} recent sessions:\n"]
        for i, s in enumerate(sessions, 1):
            sid = s["session_id"]
            ck = cache.cache_key(s["file"])
            summary = cache.get(sid, ck, "summary") or {}
            title = summary.get("title", "Untitled")
            state = summary.get("state", "")
            age = relative_time(s["mtime"], compact=True)
            lines.append(
                f"[{i}] {title}\n"
                f"    Session: {sid}\n"
                f"    Project: {s['project_dir']} | {age} | {state}\n"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error listing sessions: {e}"}]}


@tool("session_detail", "Get all indexed chunks for a specific session. Use for deep dives.", {
    "type": "object",
    "properties": {
        "session_id": {"type": "string", "description": "Session UUID to examine"},
    },
    "required": ["session_id"],
})
async def session_detail(args):
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: insights database not available"}]}
    try:
        rows = conn.execute(
            """SELECT chunk_type, timestamp, content, metadata
               FROM chunks WHERE session_id = ?
               ORDER BY timestamp LIMIT 20""",
            (args["session_id"],),
        ).fetchall()

        if not rows:
            return {"content": [{"type": "text", "text": f"No chunks found for session {args['session_id']}"}]}

        # Also get entities for this session
        entities = conn.execute(
            """SELECT entity_type, value FROM entities
               WHERE session_id = ? ORDER BY entity_type""",
            (args["session_id"],),
        ).fetchall()

        lines = [f"Session {args['session_id']} — {len(rows)} chunks:\n"]

        if entities:
            by_type: dict[str, list[str]] = {}
            for etype, val in entities:
                by_type.setdefault(etype, []).append(val)
            lines.append("Entities:")
            for etype, vals in by_type.items():
                lines.append(f"  {etype}: {', '.join(vals[:5])}")
            lines.append("")

        for i, (ctype, ts, content, meta_json) in enumerate(rows, 1):
            snippet = content[:400].replace("\n", "\n    ")
            meta = json.loads(meta_json) if meta_json else {}
            tools = meta.get("tools_used", [])
            lines.append(
                f"--- Chunk {i} [{ctype}] {ts[:16]} ---\n"
                f"    Tools: {', '.join(tools) if tools else 'none'}\n"
                f"    {snippet}\n"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    finally:
        conn.close()


@tool("hybrid_search", "Search sessions using semantic, full-text, and entity search combined with Reciprocal Rank Fusion. Best for general queries.", {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Natural language search query"},
        "limit": {"type": "integer", "description": "Max results (default 8)"},
    },
    "required": ["query"],
})
async def hybrid_search(args):
    from ..insights import rrf_search
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: insights database not available"}]}
    try:
        model = _get_cached_model()
        results = rrf_search(
            args["query"], conn, model=model,
            limit=args.get("limit", 8),
        )
        if not results:
            return {"content": [{"type": "text", "text": "No results found."}]}

        lines = [f"Found {len(results)} results (hybrid RRF):\n"]
        for i, r in enumerate(results, 1):
            snippet = r.content[:300].replace("\n", " ")
            lines.append(
                f"[{i}] score={1.0 - r.distance:.4f} | {r.chunk_type} | "
                f"{r.timestamp[:16]} | {r.project_path}\n"
                f"    Session: {r.session_id}\n"
                f"    {snippet}\n"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    finally:
        conn.close()


@tool("db_overview", "Get statistics about the indexed session database.", {
    "type": "object",
    "properties": {},
    "required": [],
})
async def db_overview(args):
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: insights database not available"}]}
    try:
        from ..insights import get_stats
        stats = get_stats(conn)
        lines = [
            "Session Intelligence Database Overview:",
            f"  Sessions indexed: {stats['sessions_indexed']}",
            f"  Total chunks: {stats['total_chunks']}",
            f"  Turn chunks: {stats['turns']}",
            f"  Subagent summaries: {stats['subagent_summaries']}",
            f"  Embedding model: {stats.get('model_name', 'unknown')}",
        ]
        if stats.get("entities_by_type"):
            lines.append("  Entities:")
            for etype, info in stats["entities_by_type"].items():
                lines.append(f"    {etype}: {info['count']} ({info['unique_values']} unique)")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    finally:
        conn.close()


# ── GitHub tools ──────────────────────────────────────────────

import subprocess


def _run_gh(args: list[str], timeout: int = 15) -> str:
    """Run a gh CLI command and return stdout. Returns error string on failure."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip()
    except FileNotFoundError:
        return "Error: gh CLI not installed"
    except subprocess.TimeoutExpired:
        return "Error: gh command timed out"


@tool("github_search_commits", "Search GitHub commit messages across repos. Great for understanding what was actually shipped and when.", {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search terms for commit messages"},
        "org": {"type": "string", "description": "GitHub org to scope search (e.g. 'greenmark-waste-solutions'). Optional."},
        "limit": {"type": "integer", "description": "Max results (default 15, max 30)"},
    },
    "required": ["query"],
})
async def github_search_commits(args):
    query = args["query"]
    limit = min(args.get("limit", 15), 30)
    gh_args = ["search", "commits", query, "--limit", str(limit),
               "--json", "repository,sha,commit"]
    org = args.get("org")
    if org:
        gh_args.extend(["--owner", org])

    raw = _run_gh(gh_args)
    if raw.startswith("Error:"):
        return {"content": [{"type": "text", "text": raw}]}

    try:
        commits = json.loads(raw)
    except json.JSONDecodeError:
        return {"content": [{"type": "text", "text": f"Failed to parse gh output: {raw[:200]}"}]}

    if not commits:
        return {"content": [{"type": "text", "text": f"No commits found for '{query}'"}]}

    lines = [f"Found {len(commits)} commits:\n"]
    for c in commits:
        repo = c.get("repository", {}).get("fullName", "?")
        msg = c.get("commit", {}).get("message", "")
        # First line of commit message only
        subject = msg.split("\n")[0]
        body = "\n".join(msg.split("\n")[1:]).strip()
        date = c.get("commit", {}).get("author", {}).get("date", "")[:10]
        sha = c.get("sha", "")[:8]
        entry = f"- [{date}] **{repo}** `{sha}` — {subject}"
        if body:
            # Include first 150 chars of body for context
            entry += f"\n  {body[:150]}"
        lines.append(entry)

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool("github_repo_commits", "List recent commits from a specific GitHub repo. Shows what changed and when.", {
    "type": "object",
    "properties": {
        "repo": {"type": "string", "description": "Full repo name: owner/repo (e.g. 'greenmark-waste-solutions/cerebro')"},
        "limit": {"type": "integer", "description": "Max commits (default 20, max 50)"},
        "since": {"type": "string", "description": "Only commits after this date (ISO 8601, e.g. '2026-01-01'). Optional."},
    },
    "required": ["repo"],
})
async def github_repo_commits(args):
    repo = args["repo"]
    limit = min(args.get("limit", 20), 50)
    since = args.get("since", "")

    jq_expr = '.[] | "\\(.commit.author.date[:10]) | \\(.sha[:8]) | \\(.commit.message)"'
    gh_args = ["api", f"repos/{repo}/commits", "--jq", jq_expr,
               "-q", f"per_page={limit}"]
    if since:
        gh_args.extend(["-f", f"since={since}"])

    raw = _run_gh(gh_args, timeout=20)
    if raw.startswith("Error:"):
        return {"content": [{"type": "text", "text": raw}]}

    if not raw.strip():
        return {"content": [{"type": "text", "text": f"No commits found in {repo}"}]}

    # Parse and format
    lines = [f"Recent commits in **{repo}**:\n"]
    for line in raw.strip().split("\n")[:limit]:
        parts = line.split(" | ", 2)
        if len(parts) == 3:
            date, sha, msg = parts
            subject = msg.split("\\n")[0] if "\\n" in msg else msg.split("\n")[0]
            body_lines = msg.replace("\\n", "\n").split("\n")[1:]
            body = " ".join(b.strip() for b in body_lines if b.strip())[:120]
            entry = f"- [{date}] `{sha}` {subject}"
            if body:
                entry += f"\n  {body}"
            lines.append(entry)
        else:
            lines.append(f"- {line}")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool("github_list_repos", "List repositories in a GitHub org to discover what repos exist.", {
    "type": "object",
    "properties": {
        "org": {"type": "string", "description": "GitHub org name (e.g. 'greenmark-waste-solutions')"},
        "limit": {"type": "integer", "description": "Max repos (default 30)"},
    },
    "required": ["org"],
})
async def github_list_repos(args):
    org = args["org"]
    limit = min(args.get("limit", 30), 50)

    raw = _run_gh(["repo", "list", org, "--limit", str(limit),
                    "--json", "name,description,updatedAt,pushedAt"])
    if raw.startswith("Error:"):
        return {"content": [{"type": "text", "text": raw}]}

    try:
        repos = json.loads(raw)
    except json.JSONDecodeError:
        return {"content": [{"type": "text", "text": f"Failed to parse: {raw[:200]}"}]}

    if not repos:
        return {"content": [{"type": "text", "text": f"No repos found in {org}"}]}

    lines = [f"Repos in **{org}** ({len(repos)}):\n"]
    for r in repos:
        desc = r.get("description", "") or "No description"
        pushed = r.get("pushedAt", "")[:10]
        lines.append(f"- **{org}/{r['name']}** — {desc} (last push: {pushed})")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


# ── Agent runner ─────────────────────────────────────────────

@tool("playbook_lookup", "Find playbooks — structured templates for briefs, reports, and status updates. ALWAYS check for a playbook before writing any structured deliverable.", {
    "type": "object",
    "properties": {
        "playbook_id": {"type": "string", "description": "Get a specific playbook by ID (e.g. 'executive-brief'). Optional."},
        "query": {"type": "string", "description": "Search playbooks by keyword. Optional."},
    },
    "required": [],
})
async def playbook_lookup(args):
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: insights database not available"}]}
    try:
        from ..playbooks import get_playbook, search_playbooks, list_playbooks

        playbook_id = args.get("playbook_id")
        query_text = args.get("query")

        if playbook_id:
            pb = get_playbook(conn, playbook_id)
            if not pb:
                return {"content": [{"type": "text", "text": f"No playbook found with ID '{playbook_id}'"}]}
            return {"content": [{"type": "text", "text": _format_playbook_full(pb)}]}

        if query_text:
            results = search_playbooks(conn, query_text)
        else:
            results = list_playbooks(conn)

        if not results:
            return {"content": [{"type": "text", "text": "No playbooks found."}]}

        lines = [f"Found {len(results)} playbook(s):\n"]
        for pb in results:
            lines.append(
                f"- **{pb.title}** (id: `{pb.id}`)\n"
                f"  {pb.description}\n"
                f"  Sections: {len(pb.sections)} | Keywords: {', '.join(pb.keywords)}\n"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    finally:
        conn.close()


def _format_playbook_full(pb) -> str:
    """Format a full playbook with all sections, data steps, and example output."""
    lines = [
        f"# Playbook: {pb.title}",
        f"ID: `{pb.id}`",
        f"Description: {pb.description}",
        f"Keywords: {', '.join(pb.keywords)}",
        "",
        "## MANDATORY INSTRUCTIONS",
        "You MUST write EVERY section below. Do NOT skip or combine sections.",
        "Each section must have substantive content — minimum 3-5 sentences per section.",
        "Use the exact section headings shown. Use markdown formatting throughout.",
        "This is a thorough briefing document, NOT a quick summary.",
        "",
        "## Sections (write ALL of these, in order):",
    ]
    for i, s in enumerate(pb.sections, 1):
        lines.append(f"\n### {i}. {s['name']}")
        lines.append(f"**What to write:** {s['description']}")
        lines.append(f"**How to write it:** {s['prompt_hint']}")

    if pb.data_steps:
        lines.append("\n## Data Gathering Steps (execute ALL of these before writing):")
        for i, step in enumerate(pb.data_steps, 1):
            lines.append(f"{i}. **{step['tool']}** — {step['purpose']}")
            if step.get("args_template"):
                lines.append(f"   Args template: {json.dumps(step['args_template'])}")

    if pb.example_output:
        lines.append("\n## EXAMPLE OUTPUT (match this length, depth, and format):")
        lines.append("The following is an example of what GOOD output looks like.")
        lines.append("Your output should match this level of detail and length.")
        lines.append("---BEGIN EXAMPLE---")
        lines.append(pb.example_output)
        lines.append("---END EXAMPLE---")

    return "\n".join(lines)


ALL_TOOLS = [
    hybrid_search, session_search, entity_search,
    list_sessions, session_detail, db_overview,
    github_search_commits, github_repo_commits, github_list_repos,
    playbook_lookup,
]

SYSTEM_PROMPT_TEMPLATE = """You are a session intelligence assistant for Claude Code. You search session transcripts AND GitHub commit history to give complete, well-sourced answers.

Available tools:

SESSION TOOLS (search conversation history):
- hybrid_search: Combined semantic + full-text + entity search with RRF (PREFERRED for general queries)
- session_search: Pure semantic/vector search over session content
- entity_search: Find sessions by file path, error class, URL, or git branch
- list_sessions: Browse recent sessions with titles
- session_detail: Deep dive into a specific session's full transcript
- db_overview: Database statistics

GITHUB TOOLS (search what was actually built/shipped):
- github_search_commits: Search commit messages across repos (great for progress briefs, changelogs)
- github_repo_commits: List recent commits from a specific repo (shows timeline of changes)
- github_list_repos: List repos in a GitHub org (discover what exists)

PLAYBOOK TOOLS (structured templates for recurring deliverables):
- playbook_lookup: Find and retrieve playbooks. ALWAYS check for a playbook before writing briefs, reports, or status updates. Playbooks define required sections and data-gathering steps.

Guidelines:
- Use hybrid_search as your default for most questions
- **For briefs, progress reports, or "what happened" questions**: ALWAYS use BOTH session search AND github_search_commits. Sessions show intent and discussions; commits show what was actually delivered.
- Use github_list_repos first if you need to discover which repos exist in an org
- Use github_repo_commits to get a chronological timeline of changes in a specific repo
- Use entity_search for specific lookups ("sessions touching /src/auth.py")
- Cite session IDs, commit SHAs, and repo names when referencing results
- If a search returns nothing useful, try different query terms or entity_search
- **If a playbook appears in "AUTO-MATCHED PLAYBOOK" below**: Follow it immediately. Execute ALL data steps, write ALL sections, match the example output's length and depth. Do NOT call playbook_lookup — the playbook is already injected.
- **For briefs, reports, or structured documents without an auto-matched playbook**: Call playbook_lookup first to check if one exists. If found, follow it completely.
- When NOT following a playbook, be concise but thorough. Summarize findings, don't dump raw data.
- Format responses with markdown: use headers, lists, code blocks, and bold text for readability.

{history_section}"""


REVIEWER_SYSTEM_PROMPT = """You are an adversarial fact-checker reviewing a brief generated from project data. Your ONLY job is to verify factual claims and flag errors.

CHECK THESE CATEGORIES (in priority order):
1. **Names & Titles/Roles** — Verify every person-role attribution. If the brief says "Michael, Director of AI", search for evidence of Michael's actual title. Person-role misattributions are the #1 source of errors. If you cannot find evidence confirming a title, flag it.
2. **Attribution of work** — Verify who did what via commit history (look at commit authors). Don't attribute work to someone without evidence.
3. **Dates & Timeline** — Cross-check dates against commit history and session timestamps. Watch for invented dates.
4. **Numbers & Metrics** — Verify counts, percentages, and quantitative claims against source data.
5. **Project/feature claims** — Verify stated capabilities, features, and status against commit evidence.

PROCESS:
- Use hybrid_search and github tools to independently verify the most important claims
- For each person mentioned, search for their name to find their actual role/title
- Do NOT verify subjective assessments or forward-looking statements
- Do NOT verify the example/template structure — only verify claims about real people, real work, real dates

OUTPUT:
If errors found, output a "## Corrections" section:
- **Claim:** [what the brief says]
- **Finding:** [what the evidence actually shows]
- **Correction:** [what it should say instead]

If no errors: "No factual errors detected in this brief."

Be thorough but efficient. A false attribution (wrong title, wrong role) is worse than a missing detail."""


def _format_history(messages: list[dict]) -> str:
    if not messages:
        return ""
    lines = ["Previous conversation:"]
    for msg in messages[-20:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long assistant responses in history
        content = msg["content"][:500] if msg["role"] == "assistant" else msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _escape(text: str) -> str:
    """HTML-escape text for SSE payloads."""
    return html.escape(text)


def _render_user_msg(message: str) -> str:
    return f'<div class="chat-msg user">{_escape(message)}</div>'


def _render_tool_call(name: str, input_args: dict, tool_use_id: str = "") -> str:
    """Render a tool call as an expandable <details> element."""
    summary = ", ".join(f"{k}={v!r}" for k, v in input_args.items())
    if len(summary) > 100:
        summary = summary[:100] + "..."
    tid = _escape(tool_use_id) if tool_use_id else ""
    return (
        f'<details class="tool-details" id="tool-{tid}">'
        f'<summary class="tool-indicator">{_escape(name)}({_escape(summary)})</summary>'
        f'<div class="tool-result-placeholder">Waiting for result...</div>'
        f'</details>'
    )


def _render_assistant_text(text: str) -> str:
    import base64
    raw_b64 = base64.b64encode(text.encode()).decode()
    return f'<div class="chat-msg assistant markdown-pending" data-raw-text="{raw_b64}"></div>'


def _render_tool_result(tool_use_id: str, result_text: str) -> str:
    """Render tool result to be injected into the matching <details> element."""
    truncated = result_text[:2000]
    if len(result_text) > 2000:
        truncated += "\n... (truncated)"
    return f'<div class="tool-result" data-tool-id="{_escape(tool_use_id)}">{_escape(truncated)}</div>'


def _render_done(result_msg: ResultMessage | None = None) -> str:
    cost = ""
    if result_msg and result_msg.total_cost_usd:
        cost = f' <small class="chat-cost">${result_msg.total_cost_usd:.4f} · {result_msg.num_turns} turns</small>'
    return f'<div class="chat-done">{cost}</div>'


def _render_error(error: str) -> str:
    return f'<div class="chat-msg error">{_escape(error)}</div>'


async def _make_prompt_stream(message: str, done_event: asyncio.Event):
    """Wrap a string prompt as an async iterable for streaming mode.

    SDK MCP servers require bidirectional communication via the control
    protocol.  When query() receives a plain string it uses --print mode
    which closes stdin immediately, breaking MCP tool responses.  An
    async iterable forces --input-format stream-json mode which keeps
    stdin open.

    After yielding the prompt, the generator waits on *done_event*.
    When the caller receives a ResultMessage it sets the event, which
    lets stream_input() close stdin so the CLI process exits cleanly.
    """
    yield {
        "type": "user",
        "message": {"role": "user", "content": message},
        "parent_tool_use_id": None,
        "session_id": "default",
    }
    # Keep stdin open until the agent conversation finishes.
    await done_event.wait()


async def _run_reviewer(
    chat_id: str,
    draft: str,
    evidence_summary: str,
    state: ChatStateManager,
) -> str:
    """Run adversarial fact-checker on a playbook-generated brief.

    Returns the review text (corrections found, or empty if clean).
    """
    # Visual separator before the review
    await state.push_event(chat_id, {
        "type": "message",
        "html": '<div class="chat-msg system" style="border-top:2px solid var(--pico-muted-border-color);padding-top:1rem;margin-top:1rem;opacity:0.85;font-size:0.9em;"><strong>Adversarial Fact-Check</strong> — Independently verifying claims against source data...</div>',
    })

    server = create_sdk_mcp_server("insights-review", tools=ALL_TOOLS)

    reviewer_prompt = (
        "Review this brief for factual errors. Verify claims against source data "
        "using the search tools. Check every name/title attribution, date, and "
        "quantitative claim.\n\n"
        f"BRIEF TO REVIEW:\n{draft}\n\n"
        f"EVIDENCE THE DRAFTER USED (verify independently — do not trust):\n"
        f"{evidence_summary}"
    )

    options = ClaudeCodeOptions(
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        model="claude-haiku-4-5-20251001",
        mcp_servers={"insights-review": server},
        permission_mode="bypassPermissions",
        max_turns=8,
        cwd=str(Path.home()),
        include_partial_messages=True,
    )

    review_text = ""
    done_event = asyncio.Event()
    got_stream_events = False

    async for msg in query(
        prompt=_make_prompt_stream(reviewer_prompt, done_event),
        options=options,
    ):
        if msg is None:
            continue

        if isinstance(msg, StreamEvent):
            event = msg.event
            event_type = event.get("type", "")

            if event_type == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "text":
                    got_stream_events = True
                    await state.push_event(chat_id, {
                        "type": "stream_start",
                        "block_index": event.get("index", 0),
                    })

            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    review_text += text
                    await state.push_event(chat_id, {
                        "type": "stream_delta",
                        "text": text,
                        "block_index": event.get("index", 0),
                    })

            elif event_type == "content_block_stop":
                await state.push_event(chat_id, {
                    "type": "stream_end",
                    "block_index": event.get("index", 0),
                })

        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    if not got_stream_events:
                        review_text += block.text
                        await state.push_event(chat_id, {
                            "type": "message",
                            "html": _render_assistant_text(block.text),
                        })
                elif isinstance(block, ToolUseBlock):
                    await state.push_event(chat_id, {
                        "type": "message",
                        "html": _render_tool_call(block.name, block.input, block.id),
                    })

        elif isinstance(msg, UserMessage):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    result_text = ""
                    if isinstance(block.content, str):
                        result_text = block.content
                    elif isinstance(block.content, list):
                        parts = []
                        for part in block.content:
                            if hasattr(part, 'text'):
                                parts.append(part.text)
                        result_text = "\n".join(parts)
                    if result_text:
                        await state.push_event(chat_id, {
                            "type": "tool_result",
                            "html": _render_tool_result(block.tool_use_id, result_text),
                        })

        elif isinstance(msg, ResultMessage):
            done_event.set()

    return review_text


async def run_agent(chat_id: str, user_message: str, state: ChatStateManager):
    """Run one agent turn. Pushes SSE events to state's event queue."""
    session = state.get_session(chat_id)
    if not session:
        return

    # Echo user message
    await state.push_event(chat_id, {
        "type": "message",
        "html": _render_user_msg(user_message),
    })

    try:
        history_section = _format_history(session.messages[:-1])  # Exclude the just-added user msg
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(history_section=history_section)

        # Auto-match: check if user message triggers a playbook
        # Tries intent patterns first (fast), then semantic similarity
        injected_playbook = None
        conn = _get_conn()
        if conn:
            try:
                from ..playbooks import match_playbook
                model = _get_cached_model()
                injected_playbook = match_playbook(conn, user_message, model=model)
            finally:
                conn.close()

        if injected_playbook:
            system_prompt += (
                "\n\n## AUTO-MATCHED PLAYBOOK\n"
                "The user's question matches the playbook below. You MUST follow it.\n"
                "Execute ALL data steps to gather information, then write ALL sections.\n"
                "Do NOT call playbook_lookup — the playbook is already here.\n\n"
                + _format_playbook_full(injected_playbook)
            )

        server = create_sdk_mcp_server("insights", tools=ALL_TOOLS)

        options = ClaudeCodeOptions(
            system_prompt=system_prompt,
            model="claude-haiku-4-5-20251001",
            mcp_servers={"insights": server},
            permission_mode="bypassPermissions",
            max_turns=10,
            cwd=str(Path.home()),
            include_partial_messages=True,
        )

        full_response = ""
        result_msg = None
        done_event = asyncio.Event()
        got_stream_events = False  # True once any StreamEvent text arrives
        playbook_used = injected_playbook is not None  # Auto-match sets this
        collected_evidence = []  # (tool_name, result_text) for reviewer
        pending_tool_names = {}  # tool_use_id -> tool_name

        async for msg in query(prompt=_make_prompt_stream(user_message, done_event), options=options):
            if msg is None:
                continue  # Skipped by lenient parser (e.g. rate_limit_event)

            if isinstance(msg, StreamEvent):
                event = msg.event
                event_type = event.get("type", "")

                if event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "text":
                        got_stream_events = True
                        await state.push_event(chat_id, {
                            "type": "stream_start",
                            "block_index": event.get("index", 0),
                        })

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        full_response += text
                        await state.push_event(chat_id, {
                            "type": "stream_delta",
                            "text": text,
                            "block_index": event.get("index", 0),
                        })

                elif event_type == "content_block_stop":
                    await state.push_event(chat_id, {
                        "type": "stream_end",
                        "block_index": event.get("index", 0),
                    })

            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        # Only render text from AssistantMessage if streaming
                        # never fired (fallback for SDK versions without streaming)
                        if not got_stream_events:
                            full_response += block.text
                            await state.push_event(chat_id, {
                                "type": "message",
                                "html": _render_assistant_text(block.text),
                            })
                    elif isinstance(block, ToolUseBlock):
                        pending_tool_names[block.id] = block.name
                        if "playbook_lookup" in block.name:
                            playbook_used = True
                        await state.push_event(chat_id, {
                            "type": "message",
                            "html": _render_tool_call(block.name, block.input, block.id),
                        })

            elif isinstance(msg, UserMessage):
                # UserMessage contains ToolResultBlocks after tool execution
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        result_text = ""
                        if isinstance(block.content, str):
                            result_text = block.content
                        elif isinstance(block.content, list):
                            parts = []
                            for part in block.content:
                                if hasattr(part, 'text'):
                                    parts.append(part.text)
                            result_text = "\n".join(parts)
                        if result_text:
                            tool_name = pending_tool_names.get(block.tool_use_id, "unknown")
                            collected_evidence.append((tool_name, result_text))
                            await state.push_event(chat_id, {
                                "type": "tool_result",
                                "html": _render_tool_result(block.tool_use_id, result_text),
                            })

            elif isinstance(msg, ResultMessage):
                result_msg = msg
                done_event.set()  # Let generator end → stdin closes → CLI exits

        if full_response:
            state.add_message(chat_id, "assistant", full_response)

        # Run adversarial fact-check if a playbook was used
        if playbook_used and full_response:
            try:
                evidence_summary = "\n\n".join(
                    f"[{name}]:\n{text[:500]}" for name, text in collected_evidence
                )
                review_text = await _run_reviewer(
                    chat_id, full_response, evidence_summary, state,
                )
                if review_text and "no factual errors" not in review_text.lower():
                    combined = (
                        full_response
                        + "\n\n---\n\n## Fact-Check Review\n\n"
                        + review_text
                    )
                    # Update stored message with corrections appended
                    if session.messages and session.messages[-1]["role"] == "assistant":
                        session.messages[-1]["content"] = combined
            except Exception as review_err:
                detail = str(review_err)
                if hasattr(review_err, 'exceptions'):
                    nested = [str(ex) for ex in review_err.exceptions]
                    detail = "; ".join(nested)
                await state.push_event(chat_id, {
                    "type": "message",
                    "html": _render_error(f"Fact-check error: {detail}"),
                })

        # Persist chat to insights DB for searchability
        try:
            from .chat_persist import save_chat
            if session.messages:
                model = _get_cached_model()
                save_chat(None, chat_id, session.messages, model=model)
        except Exception:
            pass  # Non-critical — don't break chat flow

        await state.push_event(chat_id, {
            "type": "done",
            "html": _render_done(result_msg),
        })

    except Exception as e:
        # Extract nested exceptions from TaskGroup/ExceptionGroup
        detail = str(e)
        if hasattr(e, 'exceptions'):
            nested = [str(ex) for ex in e.exceptions]
            detail = "; ".join(nested)
        await state.push_event(chat_id, {
            "type": "error",
            "html": _render_error(f"Agent error: {detail}"),
        })
