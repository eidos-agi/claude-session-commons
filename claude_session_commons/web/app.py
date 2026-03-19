"""FastAPI observability dashboard for claude-session-daemon.

Provides a read-only web UI for monitoring daemon status, browsing
indexed sessions, searching the insights database, tailing logs,
and an agentic RAG chat interface.

Entry point: claude-session-web (port 8411)
"""

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

CLAUDE_DIR = Path.home() / ".claude"
STATUS_FILE = CLAUDE_DIR / "daemon.status.json"
LOG_FILE = CLAUDE_DIR / "daemon.log"
DB_PATH = str(CLAUDE_DIR / "insights.db")

app = FastAPI(title="Session Daemon Dashboard", docs_url="/docs")

# ── Chat state ───────────────────────────────────────────────

from .chat_state import ChatStateManager

chat_state = ChatStateManager()

# ── Lazy connections ─────────────────────────────────────────

_embed_model = None
_db_initialized = False


def _get_insights_conn():
    """Open a fresh insights DB connection per call (thread-safe).

    SQLite connections can't be shared across threads, and uvicorn runs
    sync endpoints in a thread pool. So we open/close per request.
    Returns None if the DB file doesn't exist or deps are missing.
    """
    global _db_initialized
    try:
        from ..insights import get_db, init_db
        if not Path(DB_PATH).exists():
            return None
        conn = get_db(DB_PATH)
        if not _db_initialized:
            try:
                init_db(conn)
                _db_initialized = True
            except Exception:
                # init_db may fail if daemon holds write lock — that's OK,
                # the daemon already initialized the schema. Mark as done
                # so we don't retry every request.
                _db_initialized = True
        return conn
    except Exception:
        return None


def _get_embed_model():
    """Lazy-load fastembed model. Returns None if unavailable."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        return _embed_model
    except Exception:
        return None


# ── Jinja2 setup ─────────────────────────────────────────────

def _get_templates():
    """Lazy-load Jinja2Templates to avoid import-time path issues in tests."""
    from fastapi.templating import Jinja2Templates
    template_dir = Path(__file__).parent / "templates"
    return Jinja2Templates(directory=str(template_dir))


# ── Snippet extraction ────────────────────────────────────────

def _extract_snippet(content: str, query: str, window: int = 200) -> str:
    """Extract a ~200-char snippet from content centered on the best query match.

    Returns the snippet with <mark> tags around matching terms.
    """
    if not content or not query:
        return content[:window] if content else ""

    # Find the best matching position using query words
    words = [w for w in query.lower().split() if len(w) > 2]
    content_lower = content.lower()
    best_pos = 0
    best_score = -1

    for i in range(0, len(content_lower) - 10, 20):
        chunk = content_lower[i:i + window]
        score = sum(1 for w in words if w in chunk)
        if score > best_score:
            best_score = score
            best_pos = i

    # Extract window around best position
    start = max(0, best_pos - 20)
    end = min(len(content), start + window)
    snippet = content[start:end]

    # Add ellipsis if truncated
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."

    # Highlight matching words
    for w in words:
        pattern = re.compile(re.escape(w), re.IGNORECASE)
        snippet = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", snippet)

    return snippet


# ── GitHub URL resolver ──────────────────────────────────────

_github_url_cache: dict[str, str | None] = {}


def _resolve_github_url(project_path: str) -> str | None:
    """Resolve a local project path to its GitHub URL. Results are cached."""
    if project_path in _github_url_cache:
        return _github_url_cache[project_path]

    if not project_path or project_path == "web-chat":
        _github_url_cache[project_path] = None
        return None

    try:
        result = subprocess.run(
            ["git", "-C", project_path, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            _github_url_cache[project_path] = None
            return None
        url = result.stdout.strip()
        # git@github.com:org/repo.git → https://github.com/org/repo
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url[15:]
        if url.endswith(".git"):
            url = url[:-4]
        _github_url_cache[project_path] = url
        return url
    except Exception:
        _github_url_cache[project_path] = None
        return None


# ── API endpoints ────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    """Read daemon.status.json heartbeat."""
    if not STATUS_FILE.exists():
        return {"error": "no heartbeat file", "state": "unknown"}
    try:
        data = json.loads(STATUS_FILE.read_text())
        # Check if daemon process is actually alive
        pid = data.get("pid", 0)
        if pid:
            try:
                os.kill(pid, 0)
                data["process_alive"] = True
            except (ProcessLookupError, PermissionError):
                data["process_alive"] = False
        return data
    except (json.JSONDecodeError, OSError):
        return {"error": "cannot read status file", "state": "unknown"}


@app.get("/api/stats")
def api_stats():
    """Get insights DB statistics."""
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "insights database not available", "total_chunks": 0}
    try:
        from ..insights import get_stats
        return get_stats(conn)
    except Exception as e:
        return {"error": str(e), "total_chunks": 0}
    finally:
        conn.close()


@app.get("/api/logs")
def api_logs(lines: int = 50):
    """Tail daemon.log."""
    if not LOG_FILE.exists():
        return {"lines": [], "error": "no log file"}
    try:
        # Read last N lines efficiently
        with open(LOG_FILE, "rb") as f:
            # Seek from end to find last N newlines
            try:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return {"lines": []}
                # Read up to 64KB from end — enough for ~50 lines
                read_size = min(size, 65536)
                f.seek(size - read_size)
                content = f.read().decode("utf-8", errors="replace")
            except OSError:
                return {"lines": [], "error": "cannot read log"}

        all_lines = content.splitlines()
        return {"lines": all_lines[-lines:]}
    except Exception as e:
        return {"lines": [], "error": str(e)}


@app.get("/api/sessions")
def api_sessions(limit: int = 20, hours: float = 720):
    """List recent sessions with cached summaries."""
    try:
        from ..discovery import find_recent_sessions
        from ..cache import SessionCache
        from ..display import relative_time, format_size

        sessions = find_recent_sessions(hours, max_sessions=limit)
        cache = SessionCache()

        results = []
        for s in sessions:
            sid = s["session_id"]
            ck = cache.cache_key(s["file"])
            summary = cache.get(sid, ck, "summary") or {}

            results.append({
                "session_id": sid,
                "project_dir": s["project_dir"],
                "title": summary.get("title", "Untitled"),
                "state": summary.get("state", ""),
                "relative_time": relative_time(s["mtime"], compact=True),
                "size": format_size(s["size"]),
                "mtime": s["mtime"],
            })
        return {"sessions": results, "count": len(results)}
    except Exception as e:
        return {"sessions": [], "count": 0, "error": str(e)}


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    chunk_type: str | None = None


@app.post("/api/search")
def api_search(req: SearchRequest):
    """Semantic search over indexed chunks."""
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "insights database not available", "results": []}

    model = _get_embed_model()
    if model is None:
        conn.close()
        return {"error": "fastembed not available — install with: pip install fastembed", "results": []}

    try:
        from ..insights import query
        results = query(
            req.query, conn, model=model,
            limit=req.limit, chunk_type=req.chunk_type,
        )
        return {
            "results": [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "project_path": r.project_path,
                    "chunk_type": r.chunk_type,
                    "timestamp": r.timestamp,
                    "content": r.content[:500],
                    "snippet": _extract_snippet(r.content, req.query),
                    "distance": round(r.distance, 4),
                    "metadata": r.metadata,
                    "github_url": _resolve_github_url(r.project_path),
                }
                for r in results
            ],
            "count": len(results),
        }
    except Exception as e:
        return {"error": str(e), "results": []}
    finally:
        conn.close()


@app.get("/api/search/entities")
def api_entity_search(
    type: str = Query(..., description="Entity type: file_path, error_class, url, git_branch, git_commit"),
    value: str = Query(..., description="Value to search for"),
    limit: int = 20,
    exact: bool = False,
):
    """Search by entity type and value (exact or fuzzy match)."""
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "insights database not available", "results": []}

    try:
        from ..insights import query_by_entity
        results = query_by_entity(conn, type, value, limit=limit, exact=exact)
        return {
            "results": [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "project_path": r.project_path,
                    "chunk_type": r.chunk_type,
                    "timestamp": r.timestamp,
                    "content": r.content[:500],
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "count": len(results),
            "entity_type": type,
            "entity_value": value,
        }
    except Exception as e:
        return {"error": str(e), "results": []}
    finally:
        conn.close()


@app.get("/api/entities/summary")
def api_entities_summary():
    """Get a summary of all entity types and top values."""
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "insights database not available", "types": {}}

    try:
        types = {}
        for row in conn.execute(
            """SELECT entity_type, COUNT(*), COUNT(DISTINCT value)
               FROM entities GROUP BY entity_type"""
        ).fetchall():
            etype = row[0]
            types[etype] = {"count": row[1], "unique_values": row[2]}
            # Top 10 values per type
            top = conn.execute(
                """SELECT value, COUNT(*) as cnt FROM entities
                   WHERE entity_type = ? GROUP BY value
                   ORDER BY cnt DESC LIMIT 10""",
                (etype,),
            ).fetchall()
            types[etype]["top_values"] = [
                {"value": r[0], "count": r[1]} for r in top
            ]
        return {"types": types}
    except Exception as e:
        return {"error": str(e), "types": {}}
    finally:
        conn.close()


# ── Chat API endpoints ───────────────────────────────────────

@app.post("/api/chat/send")
async def chat_send(
    chat_id: str = Form(...),
    message: str = Form(...),
    agent: str = Form("rag"),
):
    """Submit a chat message. Spawns agent in background, returns immediately."""
    session = chat_state.get_session(chat_id)
    if not session:
        return {"error": "chat session not found"}
    if not message.strip():
        return {"error": "empty message"}

    chat_state.add_message(chat_id, "user", message.strip())

    if agent == "playbook":
        from .playbook_agent import run_agent as playbook_run
        asyncio.create_task(playbook_run(chat_id, message.strip(), chat_state))
    else:
        from .rag_agent import run_agent
        asyncio.create_task(run_agent(chat_id, message.strip(), chat_state))
    return {"ok": True}


@app.get("/api/chat/sse")
async def chat_sse(chat_id: str):
    """SSE stream for chat events. Consumed by HTMX sse-ext and custom EventSource."""
    async def event_gen():
        async for event in chat_state.consume_events(chat_id):
            event_type = event.get("type", "message")
            if event_type in ("stream_start", "stream_delta", "stream_end"):
                yield {"event": event_type, "data": json.dumps(event)}
            elif event_type == "tool_result":
                yield {"event": "tool_result", "data": event.get("html", "")}
            else:
                yield {"event": event_type, "data": event.get("html", "")}

    return EventSourceResponse(event_gen())


# ── Page routes ──────────────────────────────────────────────


def _chat_context(reload: str | None = None) -> dict:
    """Build shared context dict for any page that embeds a chat panel."""
    new_chat_id = chat_state.create_session()

    stats = {"sessions_indexed": 0, "total_chunks": 0}
    conn = _get_insights_conn()
    if conn:
        try:
            from ..insights import get_stats
            stats = get_stats(conn)
        except Exception:
            pass
        finally:
            conn.close()

    reload_messages = []
    reload_title = ""
    if reload:
        conn = _get_insights_conn()
        if conn:
            try:
                from .chat_persist import load_chat
                saved = load_chat(conn, reload)
                if saved:
                    reload_messages = saved["messages"]
                    reload_title = saved["title"]
                    for msg in reload_messages:
                        chat_state.add_message(new_chat_id, msg["role"], msg["content"])
            except Exception:
                pass
            finally:
                conn.close()

    return {
        "chat_id": new_chat_id,
        "stats": stats,
        "reload_messages": json.dumps(reload_messages),
        "reload_title": reload_title,
    }


@app.get("/", response_class=HTMLResponse)
def home_page(request: Request, reload: str | None = None):
    """Unified home — search on the left, chat on the right."""
    ctx = _chat_context(reload)
    ctx["page"] = "home"
    welcome = (
        f"Session Intelligence ready. {ctx['stats']['sessions_indexed']} sessions indexed, "
        f"{ctx['stats']['total_chunks']} chunks searchable. "
        "Ask me anything about your Claude Code history."
    )
    ctx["welcome_text"] = welcome
    ctx["placeholder"] = "Which sessions worked on claude-resume?"
    templates = _get_templates()
    return templates.TemplateResponse(request, "home.html", ctx)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    """Daemon monitoring dashboard."""
    templates = _get_templates()
    return templates.TemplateResponse(request, "dashboard.html", {
        "page": "dashboard",
    })


@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request):
    """Redirect to home page (search is on the home page now)."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/", status_code=302)


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request, reload: str | None = None):
    """Standalone chat page. Kept for direct links and ?reload= URLs."""
    ctx = _chat_context(reload)
    ctx["page"] = "chat"
    ctx["welcome_text"] = (
        f"Session Intelligence ready. {ctx['stats']['sessions_indexed']} sessions indexed, "
        f"{ctx['stats']['total_chunks']} chunks searchable. "
        "Ask me anything about your Claude Code history."
    )
    ctx["placeholder"] = "Which sessions worked on claude-resume?"
    templates = _get_templates()
    return templates.TemplateResponse(request, "chat.html", ctx)


# ── Chat history & actions ───────────────────────────────────

@app.get("/api/chat/history")
def api_chat_history(limit: int = 20):
    """List recent saved chats."""
    conn = _get_insights_conn()
    if conn is None:
        return {"chats": [], "error": "database not available"}
    try:
        from .chat_persist import list_recent_chats
        chats = list_recent_chats(conn, limit=limit)
        return {"chats": chats, "count": len(chats)}
    except Exception as e:
        return {"chats": [], "error": str(e)}
    finally:
        conn.close()


@app.get("/api/chat/load/{saved_chat_id}")
def api_chat_load(saved_chat_id: str):
    """Load a saved chat's messages."""
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "database not available"}
    try:
        from .chat_persist import load_chat
        chat = load_chat(conn, saved_chat_id)
        if not chat:
            return {"error": "chat not found"}
        return chat
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


@app.post("/api/launch-iterm")
def api_launch_iterm(
    session_id: str = Form(...),
    project_dir: str = Form(...),
):
    """Open iTerm2 and resume a Claude session."""
    # Sanitize inputs to prevent AppleScript injection
    safe_project_dir = project_dir.replace('"', '\\"').replace("'", "\\'")
    safe_session_id = session_id.replace('"', '\\"').replace("'", "\\'")
    cmd = f"cd {safe_project_dir} && claude --resume {safe_session_id}"
    script = (
        'tell application "iTerm2"\n'
        "    activate\n"
        "    set newWindow to (create window with default profile)\n"
        "    tell current session of newWindow\n"
        f'        write text "{cmd}"\n'
        "    end tell\n"
        "end tell"
    )
    try:
        subprocess.run(["osascript", "-e", script], timeout=10, capture_output=True)
        return {"ok": True}
    except subprocess.TimeoutExpired:
        # AppleScript may take a moment to launch iTerm — still likely succeeded
        return {"ok": True, "note": "command sent (timeout waiting for confirmation)"}
    except Exception as e:
        return {"error": str(e)}


# ── Playbooks page ────────────────────────────────────────────

@app.get("/playbooks", response_class=HTMLResponse)
def playbooks_page(request: Request):
    """Playbooks management page with embedded chat agent."""
    ctx = _chat_context()
    ctx["page"] = "playbooks"
    ctx["welcome_text"] = (
        "I can help you create, edit, or manage playbooks. "
        "Try: \"Create a playbook for weekly standup notes\" or "
        "\"Add a section to the executive brief.\""
    )
    ctx["placeholder"] = "Create a playbook for weekly standup notes..."
    ctx["agent_type"] = "playbook"
    templates = _get_templates()
    return templates.TemplateResponse(request, "playbooks.html", ctx)


# ── Playbooks CRUD API ───────────────────────────────────────

class PlaybookCreate(BaseModel):
    title: str
    description: str
    sections: list[dict]
    data_steps: list[dict] | None = None
    keywords: list[str] | None = None
    intent_patterns: list[str] | None = None


class PlaybookUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    sections: list[dict] | None = None
    data_steps: list[dict] | None = None
    keywords: list[str] | None = None
    intent_patterns: list[str] | None = None


@app.get("/api/playbooks")
def api_list_playbooks():
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "insights database not available", "playbooks": []}
    try:
        from ..playbooks import list_playbooks
        pbs = list_playbooks(conn)
        return {"playbooks": [_playbook_to_dict(pb) for pb in pbs]}
    except Exception as e:
        return {"error": str(e), "playbooks": []}
    finally:
        conn.close()


@app.get("/api/playbooks/{playbook_id}")
def api_get_playbook(playbook_id: str):
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "insights database not available"}
    try:
        from ..playbooks import get_playbook
        pb = get_playbook(conn, playbook_id)
        if not pb:
            return {"error": "playbook not found"}
        return _playbook_to_dict(pb)
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


@app.post("/api/playbooks")
def api_create_playbook(req: PlaybookCreate):
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "insights database not available"}
    try:
        from ..playbooks import create_playbook
        pb = create_playbook(
            conn, req.title, req.description, req.sections,
            data_steps=req.data_steps, keywords=req.keywords,
            intent_patterns=req.intent_patterns,
        )
        return _playbook_to_dict(pb)
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


@app.put("/api/playbooks/{playbook_id}")
def api_update_playbook(playbook_id: str, req: PlaybookUpdate):
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "insights database not available"}
    try:
        from ..playbooks import update_playbook
        pb = update_playbook(
            conn, playbook_id,
            title=req.title, description=req.description,
            sections=req.sections, data_steps=req.data_steps,
            keywords=req.keywords, intent_patterns=req.intent_patterns,
        )
        if not pb:
            return {"error": "playbook not found"}
        return _playbook_to_dict(pb)
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


@app.delete("/api/playbooks/{playbook_id}")
def api_delete_playbook(playbook_id: str):
    conn = _get_insights_conn()
    if conn is None:
        return {"error": "insights database not available"}
    try:
        from ..playbooks import delete_playbook
        ok = delete_playbook(conn, playbook_id)
        if not ok:
            return {"error": "playbook not found"}
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


def _playbook_to_dict(pb) -> dict:
    """Convert a Playbook dataclass to a JSON-serializable dict."""
    return {
        "id": pb.id,
        "title": pb.title,
        "description": pb.description,
        "sections": pb.sections,
        "data_steps": pb.data_steps,
        "keywords": pb.keywords,
        "example_output": pb.example_output,
        "intent_patterns": pb.intent_patterns,
        "created_at": pb.created_at,
        "updated_at": pb.updated_at,
    }


# ── Entry point ──────────────────────────────────────────────

def main():
    """CLI entry point: claude-session-web."""
    import uvicorn
    port = int(os.environ.get("SESSION_WEB_PORT", "8411"))
    print(f"Starting session dashboard on http://localhost:{port}")
    uvicorn.run(
        "claude_session_commons.web.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
