"""Playbook Chat Agent — CRUD tools for creating and editing playbooks.

A dedicated agent for the /playbooks page that can create, edit, list,
and delete playbooks via natural language conversation.
"""

import asyncio
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

# Reuse render helpers from rag_agent
from .rag_agent import (
    _escape,
    _render_user_msg,
    _render_tool_call,
    _render_tool_result,
    _render_assistant_text,
    _render_done,
    _render_error,
    _make_prompt_stream,
    _format_history,
)

# Reuse lenient parser patch (already applied by rag_agent import)

# ── DB helpers ────────────────────────────────────────────────

DB_PATH = str(Path.home() / ".claude" / "insights.db")


def _get_conn():
    """Open a fresh insights DB connection."""
    try:
        from ..insights import get_db, init_db
        if not Path(DB_PATH).exists():
            return None
        conn = get_db(DB_PATH)
        init_db(conn)
        return conn
    except Exception:
        return None


# ── Playbook CRUD tools ──────────────────────────────────────

@tool("list_playbooks", "List all available playbooks with their titles, descriptions, and section counts.", {
    "type": "object",
    "properties": {},
    "required": [],
})
async def list_playbooks_tool(args):
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: database not available"}]}
    try:
        from ..playbooks import list_playbooks
        pbs = list_playbooks(conn)
        if not pbs:
            return {"content": [{"type": "text", "text": "No playbooks found."}]}

        lines = [f"Found {len(pbs)} playbook(s):\n"]
        for pb in pbs:
            lines.append(
                f"- **{pb.title}** (id: `{pb.id}`)\n"
                f"  {pb.description}\n"
                f"  Sections: {len(pb.sections)} | Keywords: {', '.join(pb.keywords)}\n"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    finally:
        conn.close()


@tool("get_playbook", "Get full details of a specific playbook including all sections and data steps.", {
    "type": "object",
    "properties": {
        "playbook_id": {"type": "string", "description": "Playbook ID to retrieve"},
    },
    "required": ["playbook_id"],
})
async def get_playbook_tool(args):
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: database not available"}]}
    try:
        from ..playbooks import get_playbook
        pb = get_playbook(conn, args["playbook_id"])
        if not pb:
            return {"content": [{"type": "text", "text": f"No playbook found with ID '{args['playbook_id']}'"}]}

        lines = [
            f"# {pb.title}",
            f"ID: `{pb.id}`",
            f"Description: {pb.description}",
            f"Keywords: {', '.join(pb.keywords)}",
            "",
            "## Sections:",
        ]
        for i, s in enumerate(pb.sections, 1):
            lines.append(f"\n### {i}. {s['name']}")
            lines.append(f"Description: {s['description']}")
            lines.append(f"Prompt hint: {s['prompt_hint']}")

        if pb.data_steps:
            lines.append("\n## Data Steps:")
            for i, step in enumerate(pb.data_steps, 1):
                lines.append(f"{i}. **{step['tool']}** — {step['purpose']}")

        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    finally:
        conn.close()


@tool("create_playbook", "Create a new playbook with title, description, sections, and optional data steps.", {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Playbook title"},
        "description": {"type": "string", "description": "Brief description of the playbook's purpose"},
        "sections": {
            "type": "array",
            "description": "List of section objects with name, description, and prompt_hint",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "prompt_hint": {"type": "string"},
                },
                "required": ["name", "description", "prompt_hint"],
            },
        },
        "data_steps": {
            "type": "array",
            "description": "Optional list of data-gathering steps with tool, args_template, and purpose",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "args_template": {"type": "object"},
                    "purpose": {"type": "string"},
                },
            },
        },
        "keywords": {
            "type": "array",
            "description": "Keywords for searching this playbook",
            "items": {"type": "string"},
        },
        "intent_patterns": {
            "type": "array",
            "description": "Regex patterns that auto-trigger this playbook when a user asks a matching question (e.g. '(?i)where\\\\s+are\\\\s+we')",
            "items": {"type": "string"},
        },
    },
    "required": ["title", "description", "sections"],
})
async def create_playbook_tool(args):
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: database not available"}]}
    try:
        from ..playbooks import create_playbook
        pb = create_playbook(
            conn,
            title=args["title"],
            description=args["description"],
            sections=args["sections"],
            data_steps=args.get("data_steps"),
            keywords=args.get("keywords"),
            intent_patterns=args.get("intent_patterns"),
        )
        return {"content": [{"type": "text", "text": f"Created playbook **{pb.title}** (id: `{pb.id}`) with {len(pb.sections)} sections."}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error creating playbook: {e}"}]}
    finally:
        conn.close()


@tool("update_playbook", "Update an existing playbook's title, description, sections, data steps, or keywords.", {
    "type": "object",
    "properties": {
        "playbook_id": {"type": "string", "description": "ID of the playbook to update"},
        "title": {"type": "string", "description": "New title (optional)"},
        "description": {"type": "string", "description": "New description (optional)"},
        "sections": {
            "type": "array",
            "description": "Complete replacement sections list (optional)",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "prompt_hint": {"type": "string"},
                },
            },
        },
        "data_steps": {
            "type": "array",
            "description": "Complete replacement data steps list (optional)",
            "items": {"type": "object"},
        },
        "keywords": {
            "type": "array",
            "description": "Complete replacement keywords list (optional)",
            "items": {"type": "string"},
        },
        "intent_patterns": {
            "type": "array",
            "description": "Complete replacement intent patterns list (optional)",
            "items": {"type": "string"},
        },
    },
    "required": ["playbook_id"],
})
async def update_playbook_tool(args):
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: database not available"}]}
    try:
        from ..playbooks import update_playbook
        pb = update_playbook(
            conn,
            playbook_id=args["playbook_id"],
            title=args.get("title"),
            description=args.get("description"),
            sections=args.get("sections"),
            data_steps=args.get("data_steps"),
            keywords=args.get("keywords"),
            intent_patterns=args.get("intent_patterns"),
        )
        if not pb:
            return {"content": [{"type": "text", "text": f"No playbook found with ID '{args['playbook_id']}'"}]}
        return {"content": [{"type": "text", "text": f"Updated playbook **{pb.title}** (id: `{pb.id}`). Now has {len(pb.sections)} sections."}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error updating playbook: {e}"}]}
    finally:
        conn.close()


@tool("delete_playbook", "Delete a playbook by ID (soft delete).", {
    "type": "object",
    "properties": {
        "playbook_id": {"type": "string", "description": "ID of the playbook to delete"},
    },
    "required": ["playbook_id"],
})
async def delete_playbook_tool(args):
    conn = _get_conn()
    if not conn:
        return {"content": [{"type": "text", "text": "Error: database not available"}]}
    try:
        from ..playbooks import delete_playbook
        ok = delete_playbook(conn, args["playbook_id"])
        if not ok:
            return {"content": [{"type": "text", "text": f"No playbook found with ID '{args['playbook_id']}'"}]}
        return {"content": [{"type": "text", "text": f"Deleted playbook `{args['playbook_id']}`."}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error deleting playbook: {e}"}]}
    finally:
        conn.close()


# ── Agent runner ─────────────────────────────────────────────

ALL_TOOLS = [
    list_playbooks_tool, get_playbook_tool,
    create_playbook_tool, update_playbook_tool, delete_playbook_tool,
]

SYSTEM_PROMPT_TEMPLATE = """You are a playbook editor. You help users create and refine structured playbooks — reusable templates that define sections, data-gathering steps, and output format for recurring deliverables like executive briefs, status reports, and project plans.

Available tools:
- list_playbooks: List all existing playbooks
- get_playbook: Get full details of a specific playbook
- create_playbook: Create a new playbook
- update_playbook: Modify a playbook's sections, keywords, or data steps
- delete_playbook: Remove a playbook

When creating a playbook:
1. Ask clarifying questions about the audience and purpose if needed
2. Propose sections with descriptions and prompt hints
3. Define data_steps (which tools to call and why) when relevant
4. Create the playbook via create_playbook tool
5. Show the result and ask if they want to refine it

Each section should have:
- name: Short title (e.g. "Executive Summary")
- description: What content goes here (e.g. "2-3 sentence overview for leadership")
- prompt_hint: Instructions for the AI filling this out (e.g. "Synthesize all findings into a concise paragraph")

Data steps define which tools should be called to gather data:
- tool: Tool name (hybrid_search, github_search_commits, github_repo_commits, github_list_repos)
- args_template: Template args with {{placeholders}} like {{project_name}}, {{org}}
- purpose: Why this data is needed

Format responses with markdown. Be helpful and proactive about suggesting good playbook structures.

{history_section}"""


async def run_agent(chat_id: str, user_message: str, state: ChatStateManager):
    """Run one playbook agent turn. Pushes SSE events to state's event queue."""
    session = state.get_session(chat_id)
    if not session:
        return

    await state.push_event(chat_id, {
        "type": "message",
        "html": _render_user_msg(user_message),
    })

    try:
        history_section = _format_history(session.messages[:-1])
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(history_section=history_section)

        server = create_sdk_mcp_server("playbooks", tools=ALL_TOOLS)

        options = ClaudeCodeOptions(
            system_prompt=system_prompt,
            model="claude-haiku-4-5-20251001",
            mcp_servers={"playbooks": server},
            permission_mode="bypassPermissions",
            max_turns=10,
            cwd=str(Path.home()),
            include_partial_messages=True,
        )

        full_response = ""
        result_msg = None
        done_event = asyncio.Event()
        got_stream_events = False

        async for msg in query(prompt=_make_prompt_stream(user_message, done_event), options=options):
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
                        if not got_stream_events:
                            full_response += block.text
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
                result_msg = msg
                done_event.set()

        if full_response:
            state.add_message(chat_id, "assistant", full_response)
        await state.push_event(chat_id, {
            "type": "done",
            "html": _render_done(result_msg),
        })

    except Exception as e:
        detail = str(e)
        if hasattr(e, 'exceptions'):
            nested = [str(ex) for ex in e.exceptions]
            detail = "; ".join(nested)
        await state.push_event(chat_id, {
            "type": "error",
            "html": _render_error(f"Agent error: {detail}"),
        })
