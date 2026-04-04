"""Background daemon for pre-computing Claude session analysis.

Polls ~/.claude/projects/ for idle JSONL session files and runs
the analysis pipeline (parse, summarize, classify, index) so that
claude-resume has instant results when opened.

Also watches ~/.claude/daemon-tasks/ for on-demand requests from
the TUI (deep dives, patterns analysis, priority summarizations).
Task files are processed before the background scan each cycle.

Entry point: claude-session-daemon
"""

import json
import logging
import os
import signal
import sys
import time
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .cache import SessionCache
from .classify import get_label_deep
from .discovery import find_recent_sessions
from .git_context import get_git_context, get_git_anchor
from .parse import parse_session
from .summarize import analyze_patterns, summarize_deep, summarize_quick

# Hierarchy (optional — generates L2 project summaries)
try:
    from .hierarchy import generate_project_summary
    HAS_HIERARCHY = True
except ImportError:
    HAS_HIERARCHY = False

# Local LLM for fast window summaries (optional — requires claude-resume[local])
_local_llm_generate = None
try:
    from claude_resume.local_llm import generate as _local_llm_generate, is_available as _local_llm_ok
    if not _local_llm_ok():
        _local_llm_generate = None
except ImportError:
    pass

# Insights (optional — requires pip install -e ".[insights]")
try:
    from .insights import get_db, init_db, index_session
    HAS_INSIGHTS = True
except ImportError:
    HAS_INSIGHTS = False

# ── Constants ────────────────────────────────────────────────

CLAUDE_DIR = Path.home() / ".claude"
LOG_FILE = CLAUDE_DIR / "daemon.log"
PID_FILE = CLAUDE_DIR / "session-daemon.pid"
STATUS_FILE = CLAUDE_DIR / "daemon.status.json"
RESUME_CACHE_DIR = CLAUDE_DIR / "resume-summaries"
TASK_DIR = CLAUDE_DIR / "daemon-tasks"
INSIGHTS_SKIP_FILE = CLAUDE_DIR / "insights-skip.json"
HUD_SOCKET_PATH = "/tmp/resume-hud.sock"

IDLE_THRESHOLD_SECS = 300       # 5 minutes — don't process active sessions
POLL_INTERVAL_SECS = 300        # 5 minutes between scans
TASK_POLL_SECS = 2              # check for TUI task requests every 2s
RATE_LIMIT_SECS = 3             # pause between sessions (on top of claude -p time)
LOOKBACK_HOURS = 720            # 30 days — only process recent sessions
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
LOG_BACKUP_COUNT = 3
DAEMON_VERSION = "1.2.0"
INSIGHTS_MAX_FAILURES = 3       # skip session after N consecutive indexing failures


# ── LLM Queue ────────────────────────────────────────────────
# Single-threaded summarization queue. One inference at a time.
# Callers submit work via llm_queue_submit(), results arrive in cache.
# Supports any local model (Gemma today, Phi-3 tomorrow).

import queue as _queue_mod

_llm_queue: _queue_mod.Queue = _queue_mod.Queue()
_llm_thread: threading.Thread | None = None
_llm_shutdown = threading.Event()


def _llm_worker(logger: logging.Logger):
    """Drain the LLM queue forever. One inference at a time. Exits on poison pill."""
    while not _llm_shutdown.is_set():
        try:
            item = _llm_queue.get(timeout=2)
        except _queue_mod.Empty:
            continue

        if item is None:  # poison pill
            break

        kind = item.get("kind")
        try:
            if kind == "window_summary":
                _llm_do_window_summary(item, logger)
            elif kind == "window_summaries_batch":
                _llm_do_window_summaries_batch(item, logger)
            else:
                logger.warning("LLM queue: unknown kind %r", kind)
        except Exception as e:
            logger.warning("LLM queue error [%s]: %s", kind, e)
        finally:
            _llm_queue.task_done()


LLM_MAX_INPUT_CHARS = 4000  # ~1K tokens for Gemma 2B — safe under 8K context window


def _cap_context(text: str, max_chars: int = LLM_MAX_INPUT_CHARS) -> str:
    """Cap context to max_chars. For long text, keep first 1K + last 3K
    so the model sees both the start and end of the time window."""
    if not text or len(text) <= max_chars:
        return text
    head = max_chars // 4       # 1K chars from start
    tail = max_chars - head     # 3K chars from end
    return text[:head] + "\n\n[... middle truncated ...]\n\n" + text[-tail:]


MAPREDUCE_CHUNK_CHARS = 1300  # ~325 tokens per chunk — fast inference per pass
MAPREDUCE_THRESHOLD = 2500    # contexts larger than this get map-reduced


def _llm_infer(prompt: str, max_tokens: int, logger: logging.Logger) -> str:
    """Run a single LLM inference. Gemma first, haiku fallback. Returns raw output."""
    import subprocess

    output = ""
    if _local_llm_generate is not None:
        try:
            output = _local_llm_generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.debug("Local LLM failed: %s", e)

    if not output or len(output.strip()) < 3:
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", "haiku", prompt],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "cli"},
            )
            output = result.stdout.strip()
        except Exception as e:
            logger.debug("Haiku fallback failed: %s", e)
            output = ""

    return output.strip()


def _llm_do_window_summary(item: dict, logger: logging.Logger):
    """Generate a single window summary (5m, 30m, or 2h).
    Uses map-reduce for large contexts: chunk → summarize each → summarize summaries."""

    key = item["key"]           # "5m", "30m", "2h"
    raw_context = item["context"] or ""
    sid = item["session_id"]
    file_path = item["file"]
    caches = item["caches"]
    callback = item.get("callback")  # optional notify function

    label_map = {"5m": "LAST 5 MINUTES", "30m": "LAST 30 MINUTES", "2h": "LAST 2 HOURS"}
    label = label_map.get(key, key)

    if not raw_context.strip():
        summary = "no activity"
    elif len(raw_context) > MAPREDUCE_THRESHOLD:
        # ── Map-Reduce: chunk → summarize each → reduce ──
        chunks = []
        for i in range(0, len(raw_context), MAPREDUCE_CHUNK_CHARS):
            chunks.append(raw_context[i:i + MAPREDUCE_CHUNK_CHARS])

        chunk_summaries = []
        for ci, chunk in enumerate(chunks):
            prompt = f"""Summarize this chunk ({ci+1}/{len(chunks)}) of a Claude Code session's {label}.
ONE sentence, max 15 words. Focus on the WHAT — name the feature, bug, or file.

{chunk}
"""
            out = _llm_infer(prompt, max_tokens=40, logger=logger)
            if out:
                chunk_summaries.append(out.split("\n")[0].strip())

        if chunk_summaries:
            # Reduce: merge chunk summaries into one
            combined = "\n".join(f"- {s}" for s in chunk_summaries)
            reduce_prompt = f"""These are summaries of consecutive chunks from a Claude Code session's {label}.
Merge into ONE sentence (max 15 words). Focus on the main activity.

{combined}
"""
            summary = _llm_infer(reduce_prompt, max_tokens=40, logger=logger)
            summary = summary.split("\n")[0].strip() if summary else "summary failed"
        else:
            summary = "summary failed"

        logger.info("LLM MAP-REDUCE: %s/%s -> %d chunks -> %s", sid[:8], key, len(chunks), summary[:40])
    else:
        # ── Single pass: small context ──
        context = _cap_context(raw_context)
        prompt = f"""Summarize what was happening in a Claude Code session during the {label}.
Write ONE sentence (max 15 words). Focus on the WHAT — name the feature, bug, or file.
If no activity, say "no activity". Return ONLY the summary, no prefix.

--- {label} ---
{context}
"""
        summary = _llm_infer(prompt, max_tokens=60, logger=logger)
        summary = summary.split("\n")[0].strip() if summary else "summary failed"

    # Clean prefix echoes
    for prefix in ("5m:", "30m:", "2h:"):
        if summary.lower().startswith(prefix):
            summary = summary[len(prefix):].strip()
    if not summary:
        summary = "no activity"

    # Merge into cache
    for cache in caches:
        ck = cache.cache_key(file_path)
        existing = cache.get(sid, ck, "window_summaries")
        if not existing or not isinstance(existing, dict):
            existing = {}
        existing[key] = summary
        cache.set(sid, ck, "window_summaries", existing)

    logger.info("LLM OK: %s/%s -> %s", sid[:8], key, summary[:40])
    if callback:
        try:
            callback(sid, key, summary)
        except Exception:
            pass


def _llm_do_window_summaries_batch(item: dict, logger: logging.Logger):
    """Generate all 3 window summaries sequentially for one session."""
    contexts = item["contexts"]  # {"5m": "...", "30m": "...", "2h": "..."}
    for key in ("5m", "30m", "2h"):
        ctx = contexts.get(key, "")
        sub = {**item, "kind": "window_summary", "key": key, "context": ctx}
        _llm_do_window_summary(sub, logger)


def llm_queue_submit(item: dict):
    """Submit work to the LLM queue. Non-blocking."""
    _llm_queue.put(item)


def llm_queue_depth() -> int:
    return _llm_queue.qsize()


def _start_llm_worker(logger: logging.Logger):
    """Start the LLM worker thread (idempotent)."""
    global _llm_thread
    if _llm_thread is not None and _llm_thread.is_alive():
        return
    _llm_shutdown.clear()
    _llm_thread = threading.Thread(target=_llm_worker, args=(logger,), daemon=True, name="llm-worker")
    _llm_thread.start()
    logger.info("LLM worker started (model=%s)", "gemma-2b" if _local_llm_generate else "haiku-fallback")


def _stop_llm_worker():
    """Signal the worker to stop and wait."""
    _llm_shutdown.set()
    _llm_queue.put(None)  # poison pill
    if _llm_thread is not None:
        _llm_thread.join(timeout=10)


# ── Logging ──────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    """Configure rotating file logger + stderr."""
    CLAUDE_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("session-daemon")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ── PID management ───────────────────────────────────────────

def _check_already_running() -> bool:
    """Return True if another instance is already running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0 = check if alive
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return False


def _write_pid():
    CLAUDE_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _remove_pid():
    PID_FILE.unlink(missing_ok=True)


# ── Status heartbeat ────────────────────────────────────────

_DAEMON_STATUS = {
    "pid": 0,
    "version": DAEMON_VERSION,
    "started_at": "",
    "last_run": "",
    "last_success": "",
    "last_error": "",
    "sessions_processed": 0,
    "sessions_failed": 0,
    "tasks_processed": 0,
    "queue_depth": 0,
    "insights_enabled": False,
    "insights_chunks": 0,
    "state": "starting",
}


def _write_status(status: dict | None = None):
    """Write daemon.status.json atomically. Uses module-level _DAEMON_STATUS if none provided."""
    data = status or _DAEMON_STATUS
    tmp = STATUS_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(STATUS_FILE)
    except OSError:
        pass  # Best-effort — don't crash the daemon over status writes


# ── Insights skip list ───────────────────────────────────────

# Track consecutive failures per session: {session_id: count}
_insights_fail_counts: dict[str, int] = {}
_insights_skip_set: set[str] = set()


def _load_skip_list():
    """Load persistent skip list of sessions that consistently fail indexing."""
    global _insights_skip_set
    if INSIGHTS_SKIP_FILE.exists():
        try:
            data = json.loads(INSIGHTS_SKIP_FILE.read_text())
            _insights_skip_set = set(data.get("skip", []))
        except Exception:
            _insights_skip_set = set()


def _save_skip_list():
    """Persist skip list to disk."""
    try:
        INSIGHTS_SKIP_FILE.write_text(json.dumps(
            {"skip": sorted(_insights_skip_set)}, indent=2,
        ))
    except OSError:
        pass


def _record_insights_failure(session_id: str, logger: logging.Logger):
    """Record an indexing failure. After INSIGHTS_MAX_FAILURES, add to skip list."""
    _insights_fail_counts[session_id] = _insights_fail_counts.get(session_id, 0) + 1
    if _insights_fail_counts[session_id] >= INSIGHTS_MAX_FAILURES:
        if session_id not in _insights_skip_set:
            _insights_skip_set.add(session_id)
            _save_skip_list()
            logger.info("INSIGHTS SKIP: %s... added to skip list after %d failures",
                        session_id[:8], INSIGHTS_MAX_FAILURES)


def _should_skip_insights(session_id: str) -> bool:
    """Check if a session is in the skip list."""
    return session_id in _insights_skip_set


# ── Session filtering ────────────────────────────────────────

def _find_idle_uncached(
    caches: list[SessionCache],
    logger: logging.Logger,
) -> list[dict]:
    """Find sessions that are idle (10min+) and missing from any cache."""
    now = time.time()
    cutoff = now - IDLE_THRESHOLD_SECS
    all_sessions = find_recent_sessions(LOOKBACK_HOURS, max_sessions=0)

    candidates = []
    for s in all_sessions:
        if s["mtime"] > cutoff:
            continue  # still active, skip

        all_cached = True
        for cache in caches:
            ck = cache.cache_key(s["file"])
            if not cache.get(s["session_id"], ck, "summary"):
                all_cached = False
                break

        if not all_cached:
            candidates.append(s)

    return candidates


def _find_insights_backlog(
    caches: list[SessionCache],
    insights_conn,
    logger: logging.Logger,
    batch_size: int = 50,
) -> list[dict]:
    """Find sessions that are cached but NOT in the insights DB.

    These were processed for quick summaries but never made it into
    the semantic search index (insights). Returns up to batch_size.
    """
    if insights_conn is None:
        return []

    from .insights import is_indexed

    now = time.time()
    cutoff = now - IDLE_THRESHOLD_SECS
    all_sessions = find_recent_sessions(LOOKBACK_HOURS, max_sessions=0)

    backlog = []
    for s in all_sessions:
        if s["mtime"] > cutoff:
            continue

        # Must already be cached (summary done)
        any_cached = False
        for cache in caches:
            ck = cache.cache_key(s["file"])
            if cache.get(s["session_id"], ck, "summary"):
                any_cached = True
                break

        if not any_cached:
            continue  # not cached yet — let _find_idle_uncached handle it

        # Check if insights indexed (and not on skip list)
        if not is_indexed(insights_conn, s["session_id"]) and not _should_skip_insights(s["session_id"]):
            backlog.append(s)
            if len(backlog) >= batch_size:
                break

    return backlog


# ── Window summaries + scoring ───────────────────────────────

def _compute_window_summaries(
    session: dict,
    caches: list[SessionCache],
    logger: logging.Logger,
) -> bool:
    """Generate ML time-window summaries (5m/30m/2h) for a session.
    Submits to the LLM queue — non-blocking, one window at a time."""
    from datetime import datetime

    sid = session["session_id"]
    file_path = session["file"]

    # Check if already cached
    for cache in caches:
        ck = cache.cache_key(file_path)
        existing = cache.get(sid, ck, "window_summaries")
        if existing and isinstance(existing, dict) and "5m" in existing:
            return True  # already done

    # Extract conversation context per window
    WINDOWS = {"5m": 5, "30m": 30, "2h": 120}
    raw: dict[str, list[str]] = {k: [] for k in WINDOWS}

    try:
        with open(file_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_from = max(0, size - 524288)
            f.seek(read_from)
            chunk = f.read().decode("utf-8", errors="replace")
            lines = chunk.strip().split("\n")
    except OSError:
        return False

    last_ts = None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            ts = entry.get("timestamp")
            if ts:
                last_ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                break
        except (json.JSONDecodeError, ValueError):
            continue

    if last_ts is None:
        return False

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = entry.get("timestamp")
        if not ts:
            continue
        try:
            entry_ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        age_min = (last_ts - entry_ts) / 60
        entry_type = entry.get("type", "")

        for key, minutes in WINDOWS.items():
            if age_min > minutes:
                continue
            if entry_type == "user":
                msg = entry.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                text = ""
                if isinstance(content, str) and len(content) > 5:
                    text = content[:300]
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")[:300]
                            break
                if text:
                    raw[key].append(f"USER: {text}")
            elif entry_type == "assistant":
                msg = entry.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                text = ""
                if isinstance(content, str):
                    text = content[:300]
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")[:300]
                            break
                if text:
                    raw[key].append(f"ASSISTANT: {text}")
            elif entry_type == "tool_use":
                name = entry.get("tool_name", entry.get("name", "?"))
                raw[key].append(f"TOOL: {name}")

    # Build prompt
    contexts = {}
    for key in WINDOWS:
        entries = raw[key][-30:]
        text = "\n".join(entries)
        contexts[key] = text[-3000:] if len(text) > 3000 else text

    if not any(contexts.values()):
        summaries = {"5m": "no activity", "30m": "no activity", "2h": "no activity"}
        for cache in caches:
            ck = cache.cache_key(file_path)
            cache.set(sid, ck, "window_summaries", summaries)
        return True

    # Submit to the LLM queue — processes one window at a time, no blocking
    llm_queue_submit({
        "kind": "window_summaries_batch",
        "session_id": sid,
        "file": file_path,
        "contexts": contexts,
        "caches": caches,
    })

    logger.info("WINDOWS QUEUED: %s... (queue depth: %d)", sid[:8], llm_queue_depth())
    return True


def _compute_active_time(
    session: dict,
    caches: list[SessionCache],
    gap_threshold: float = 120.0,
) -> dict:
    """Compute active time by summing inter-entry gaps < threshold (default 120s).

    Returns {"active_seconds": int, "total_seconds": int, "focus_pct": float}
    Cached as "active_time" key.
    """
    from datetime import datetime

    sid = session["session_id"]
    cache = caches[0]
    ck = cache.cache_key(session["file"])

    existing = cache.get(sid, ck, "active_time")
    if existing and isinstance(existing, dict):
        return existing

    try:
        with open(session["file"], "rb") as f:
            data = f.read().decode("utf-8", errors="replace")
            lines = data.strip().split("\n")
    except OSError:
        return {"active_seconds": 0, "total_seconds": 0, "focus_pct": 0.0}

    timestamps = []
    for line in lines:
        try:
            entry = json.loads(line)
            ts = entry.get("timestamp")
            if ts:
                timestamps.append(
                    datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                )
        except (json.JSONDecodeError, ValueError):
            continue

    if len(timestamps) < 2:
        result = {"active_seconds": 0, "total_seconds": 0, "focus_pct": 0.0}
        for c in caches:
            c.set(sid, c.cache_key(session["file"]), "active_time", result)
        return result

    timestamps.sort()
    total = timestamps[-1] - timestamps[0]
    active = sum(
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
        if (timestamps[i + 1] - timestamps[i]) <= gap_threshold
    )

    focus_pct = round((active / total * 100) if total > 0 else 0, 1)
    result = {
        "active_seconds": round(active),
        "total_seconds": round(total),
        "focus_pct": focus_pct,
    }

    for c in caches:
        c.set(sid, c.cache_key(session["file"]), "active_time", result)
    return result


def _compute_resumability_score(
    session: dict,
    caches: list[SessionCache],
) -> float:
    """Compute and cache resumability score (0-100)."""
    import math

    sid = session["session_id"]
    cache = caches[0]  # primary cache
    ck = cache.cache_key(session["file"])

    # Check existing
    existing = cache.get(sid, ck, "resumability_score")
    if existing is not None:
        return existing

    # Bookmark overrides
    bookmark = cache.get(sid, ck, "bookmark")
    lifecycle = None
    if bookmark and isinstance(bookmark, dict):
        lifecycle = bookmark.get("lifecycle_state")
    if lifecycle == "done":
        for c in caches:
            c.set(sid, c.cache_key(session["file"]), "resumability_score", 5.0)
        return 5.0

    # Engagement (35)
    stats = cache.get(sid, ck, "stats")
    engagement = 0.0
    if stats and isinstance(stats, dict):
        size = stats.get("file_size", session.get("size", 0))
        if size > 0:
            engagement += min(35, 5 * math.log10(max(size, 1000) / 1000) + 5)
        if stats.get("user_messages", 0) > 20:
            engagement += 5
        if stats.get("tool_uses", 0) > 50:
            engagement += 5
    else:
        size = session.get("size", 0)
        if size > 0:
            engagement += min(35, 5 * math.log10(max(size, 1000) / 1000) + 5)
    engagement = min(engagement, 35)

    # Recency (30) — 4hr half-life
    age_hours = (time.time() - session["mtime"]) / 3600
    recency = 30.0 * math.exp(-0.693 * age_hours / 4)

    # Unfinished (25)
    unfinished_map = {"user": 25, "progress": 22, "tool_result": 20, "assistant": 10, "summary": 5}
    unfinished = unfinished_map.get(session.get("last_entry_type", ""), 8)
    summary = cache.get(sid, ck, "summary")
    if summary and isinstance(summary, dict):
        state = summary.get("state", "")
        if state and any(w in state.lower() for w in ("done", "completed", "finished")):
            unfinished = max(unfinished - 10, 0)

    # Classification (10)
    classification = 5.0
    if stats and isinstance(stats, dict):
        cls = stats.get("classification", "pending")
        classification = 10 if cls == "interactive" else (2 if cls == "automated" else 5)

    score = min(round(engagement + recency + unfinished + classification, 1), 100)

    if lifecycle == "blocked":
        score = max(score, 60)
    elif lifecycle == "paused":
        score = max(score, 40)

    for c in caches:
        c.set(sid, c.cache_key(session["file"]), "resumability_score", score)
    return score


# ── Analysis pipeline ────────────────────────────────────────

def _analyze_session(
    session: dict,
    caches: list[SessionCache],
    logger: logging.Logger,
    insights_conn=None,
    insights_model=None,
) -> bool:
    """Run quick summary + classify on one session. Returns True on success."""
    sid = session["session_id"]
    project_dir = session["project_dir"]

    try:
        context, search_text = parse_session(session["file"])
        git = get_git_context(project_dir)
        summary = summarize_quick(context, project_dir, git)

        full_search = (search_text + f" {project_dir} {sid}").lower()

        for cache in caches:
            ck = cache.cache_key(session["file"])
            cache.set(sid, ck, "summary", summary)
            cache.set(sid, ck, "search_text", full_search)

        # Deep classification (may call claude -p for uncertain cases)
        for cache in caches:
            get_label_deep(session["file"], cache)

        # Insights indexing (optional — embed chunks for semantic search)
        if insights_conn is not None and not _should_skip_insights(sid):
            try:
                git_anchor = get_git_anchor(project_dir)
                turns, subagents = index_session(
                    str(session["file"]), insights_conn,
                    model=insights_model,
                    session_id=sid,
                    project_path=project_dir,
                    git_anchor=git_anchor if git_anchor.get("branch") else None,
                )
                if turns or subagents:
                    logger.info("INSIGHTS: %s... -> %d turns, %d subagents",
                                sid[:8], turns, subagents)
            except Exception as ie:
                logger.warning("INSIGHTS FAIL: %s... -> %s", sid[:8], ie)
                _record_insights_failure(sid, logger)

        # Window summaries (ML — local Gemma 2B or haiku fallback)
        try:
            _compute_window_summaries(session, caches, logger)
        except Exception as we:
            logger.debug("Window summaries skipped for %s: %s", sid[:8], we)

        # Resumability score
        try:
            _compute_resumability_score(session, caches)
        except Exception as se:
            logger.debug("Score skipped for %s: %s", sid[:8], se)

        # Active time / focus metric
        try:
            _compute_active_time(session, caches)
        except Exception as ae:
            logger.debug("Active time skipped for %s: %s", sid[:8], ae)

        title = summary.get("title", "?")
        logger.info("OK: %s... -> %s", sid[:8], title)
        return True

    except Exception as e:
        logger.warning("FAIL: %s... -> %s", sid[:8], e)
        return False


def _index_session_only(
    session: dict,
    logger: logging.Logger,
    insights_conn=None,
    insights_model=None,
) -> bool:
    """Index a session into insights DB only (skip summary/classify).

    Used for backfilling sessions that are already cached but missing
    from the semantic search index.
    """
    if insights_conn is None:
        return False

    sid = session["session_id"]
    project_dir = session["project_dir"]

    if _should_skip_insights(sid):
        return False

    try:
        git_anchor = get_git_anchor(project_dir)
        turns, subagents = index_session(
            str(session["file"]), insights_conn,
            model=insights_model,
            session_id=sid,
            project_path=project_dir,
            git_anchor=git_anchor if git_anchor.get("branch") else None,
        )
        if turns or subagents:
            logger.info("INSIGHTS BACKFILL: %s... -> %d turns, %d subagents",
                        sid[:8], turns, subagents)
            return True
        return False
    except Exception as ie:
        logger.warning("INSIGHTS BACKFILL FAIL: %s... -> %s", sid[:8], ie)
        _record_insights_failure(sid, logger)
        return False


# ── On-demand task processing (TUI delegation) ──────────────

def _process_task_files(
    caches: list[SessionCache],
    shutdown: threading.Event,
    logger: logging.Logger,
) -> int:
    """Process task files from TUI. Returns count processed."""
    if not TASK_DIR.exists():
        return 0

    task_files = sorted(TASK_DIR.glob("*.json"), reverse=True)  # newest first
    if not task_files:
        return 0

    logger.info("Processing %d task request(s) from TUI", len(task_files))
    count = 0

    for tf in task_files:
        if shutdown.is_set():
            break
        try:
            task = json.loads(tf.read_text())
        except (json.JSONDecodeError, OSError):
            tf.unlink(missing_ok=True)
            continue

        kind = task.get("kind", "summarize")
        sid = task.get("session_id", "")
        file_path = Path(task.get("file", ""))
        project_dir = task.get("project_dir", "")
        quick_summary = task.get("quick_summary")

        try:
            if kind == "summarize":
                context, search_text = parse_session(file_path)
                git = get_git_context(project_dir)
                summary = summarize_quick(context, project_dir, git)
                full_search = (search_text + f" {project_dir} {sid}").lower()
                for cache in caches:
                    ck = cache.cache_key(file_path)
                    cache.set(sid, ck, "summary", summary)
                    cache.set(sid, ck, "search_text", full_search)
                for cache in caches:
                    get_label_deep(file_path, cache)
                logger.info("TASK OK [%s]: %s... -> %s", kind, sid[:8],
                            summary.get("title", "?"))

            elif kind == "deep":
                context, _ = parse_session(file_path, deep=True)
                git = get_git_context(project_dir)
                quick = quick_summary or {"title": "Unknown", "state": "", "files": []}
                deep = summarize_deep(context, project_dir, quick, git)
                for cache in caches:
                    ck = cache.cache_key(file_path)
                    cache.set(sid, ck, "deep_summary", deep)
                logger.info("TASK OK [%s]: %s...", kind, sid[:8])

            elif kind == "patterns":
                context, _ = parse_session(file_path, deep=True)
                quick = quick_summary or {"title": "Unknown", "state": "", "files": []}
                patterns = analyze_patterns(context, project_dir, quick)
                for cache in caches:
                    ck = cache.cache_key(file_path)
                    cache.set(sid, ck, "patterns", patterns)
                logger.info("TASK OK [%s]: %s...", kind, sid[:8])

            elif kind == "window_summaries":
                _compute_window_summaries(
                    {"session_id": sid, "file": file_path, "project_dir": project_dir},
                    caches, logger,
                )
                logger.info("TASK OK [%s]: %s...", kind, sid[:8])

            elif kind == "active_time":
                _compute_active_time(
                    {"session_id": sid, "file": file_path, "project_dir": project_dir},
                    caches,
                )
                logger.info("TASK OK [%s]: %s...", kind, sid[:8])

            elif kind == "score":
                session_dict = {
                    "session_id": sid, "file": file_path,
                    "project_dir": project_dir,
                    "mtime": file_path.stat().st_mtime if file_path.exists() else 0,
                    "size": file_path.stat().st_size if file_path.exists() else 0,
                }
                _compute_resumability_score(session_dict, caches)
                logger.info("TASK OK [%s]: %s...", kind, sid[:8])

            count += 1

        except Exception as e:
            logger.warning("TASK FAIL [%s]: %s... -> %s", kind, sid[:8], e)

        tf.unlink(missing_ok=True)

    return count


# ── Web child (claude-resume-duet) ───────────────────────────
# The main daemon owns the lifecycle of the claude-resume-duet-serve
# subprocess. Clean separation: no shared memory, just HTTP pings.

import subprocess as _subprocess
import shutil as _shutil
import urllib.request as _urllib_request

CRD_PORT = 8412
CRD_HEALTH_URL = f"http://localhost:{CRD_PORT}/health"
CRD_NOTIFY_URL = f"http://localhost:{CRD_PORT}/notify"
CRD_BACKOFF_SECS = [5, 10, 30, 60]  # restart delay sequence

_web_child: "_subprocess.Popen | None" = None
_web_restart_count = 0


def _crd_binary() -> str | None:
    """Return path to claude-resume-duet-serve, or None if not installed."""
    return _shutil.which("claude-resume-duet-serve")


def _crd_healthy() -> bool:
    """Quick health check — returns True if the web child responds."""
    try:
        with _urllib_request.urlopen(CRD_HEALTH_URL, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _start_web_child(logger: logging.Logger) -> bool:
    """Spawn claude-resume-duet-serve. Returns True on success."""
    global _web_child, _web_restart_count
    binary = _crd_binary()
    if not binary:
        return False  # not installed — skip silently
    try:
        log_out = open(CLAUDE_DIR / "crd-stdout.log", "a")
        log_err = open(CLAUDE_DIR / "crd-stderr.log", "a")
        _web_child = _subprocess.Popen(
            [binary],
            stdout=log_out,
            stderr=log_err,
            close_fds=True,
        )
        logger.info("Web child started (pid=%d, port=%d)", _web_child.pid, CRD_PORT)
        _web_restart_count += 1
        return True
    except Exception as e:
        logger.warning("Failed to start web child: %s", e)
        _web_child = None
        return False


def _check_web_child(logger: logging.Logger):
    """Restart the web child if it has died or become unresponsive. Called each poll cycle."""
    global _web_child
    if _web_child is None:
        _start_web_child(logger)
        return
    # Process exited OR health check fails → restart
    if _web_child.poll() is not None or not _crd_healthy():
        rc = _web_child.returncode
        delay = CRD_BACKOFF_SECS[min(_web_restart_count, len(CRD_BACKOFF_SECS) - 1)]
        logger.warning("Web child exited (rc=%d), restarting in %ds", rc, delay)
        _web_child = None
        time.sleep(delay)
        _start_web_child(logger)


def _notify_web_child():
    """Ping the web child after a scan so it refreshes its session list."""
    try:
        req = _urllib_request.Request(CRD_NOTIFY_URL, method="POST", data=b"")
        with _urllib_request.urlopen(req, timeout=2):
            pass
    except Exception:
        pass  # fire-and-forget — web child may be starting up


def _stop_web_child(logger: logging.Logger):
    """Terminate the web child on daemon shutdown."""
    global _web_child
    if _web_child is None:
        return
    try:
        _web_child.terminate()
        _web_child.wait(timeout=5)
        logger.info("Web child stopped")
    except Exception:
        try:
            _web_child.kill()
        except Exception:
            pass
    _web_child = None


# ── HUD progress bridge ────────────────────────────────────

import select as _select_mod
import socket as _socket_mod
import subprocess as _subprocess_mod

_hud_fifo = None  # type: ignore[assignment]  # file handle for FIFO writes


_hud_pid: int | None = None


def _hud_process_alive() -> bool:
    """Check if a HUD process with a listening socket exists."""
    pid_path = Path("/tmp/resume-hud.pid")
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def _start_hud_bridge(shutdown: threading.Event, logger: logging.Logger):
    """Start a Unix socket listener that bridges MCP progress events to the HUD.

    MCP tools connect to the socket and write JSON-lines. This thread
    reads them and forwards to the HUD subprocess's stdin (which the
    HUD renders via WKWebView).
    """
    sock_path = Path(HUD_SOCKET_PATH)
    sock_path.unlink(missing_ok=True)

    server = _socket_mod.socket(_socket_mod.AF_UNIX, _socket_mod.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(5)
    server.setblocking(False)
    logger.info("HUD bridge listening on %s", HUD_SOCKET_PATH)

    # The bridge acts as a fan-out relay: MCP tools connect as writers,
    # the HUD process connects as a reader. Both sides use the same socket.
    # The HUD is spawned by the MCP server (which has GUI access), not the daemon.
    clients: list[_socket_mod.socket] = []
    buffers: dict[_socket_mod.socket, bytes] = {}
    hud_writers: list[_socket_mod.socket] = []  # connections from HUD for reading

    while not shutdown.is_set():
        readable = [server] + clients
        try:
            ready, _, _ = _select_mod.select(readable, [], [], 1.0)
        except (ValueError, OSError):
            break

        for sock in ready:
            if sock is server:
                conn, _ = server.accept()
                conn.setblocking(False)
                clients.append(conn)
                buffers[conn] = b""
            else:
                try:
                    data = sock.recv(4096)
                except (ConnectionResetError, OSError):
                    data = b""

                if not data:
                    clients.remove(sock)
                    del buffers[sock]
                    sock.close()
                    continue

                buffers[sock] += data
                while b"\n" in buffers[sock]:
                    line, buffers[sock] = buffers[sock].split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    # Fan out to ALL other connected clients (HUD reads from here)
                    dead = []
                    for c in clients:
                        if c is sock:
                            continue
                        try:
                            c.sendall(line + b"\n")
                        except (BrokenPipeError, OSError):
                            dead.append(c)
                    for d in dead:
                        clients.remove(d)
                        del buffers[d]
                        d.close()

    server.close()
    sock_path.unlink(missing_ok=True)
    logger.info("HUD bridge stopped")


# ── Main loop ────────────────────────────────────────────────

def _run_loop(shutdown: threading.Event, logger: logging.Logger):
    """Poll-and-process loop. Runs until shutdown event is set."""
    from datetime import datetime, timezone

    caches = [
        SessionCache(),
        SessionCache(RESUME_CACHE_DIR),
    ]

    # Initialize status heartbeat
    now_iso = datetime.now(timezone.utc).isoformat()
    _DAEMON_STATUS["pid"] = os.getpid()
    _DAEMON_STATUS["started_at"] = now_iso
    _DAEMON_STATUS["state"] = "starting"
    _write_status()

    # Initialize insights (optional — graceful if deps missing)
    insights_conn = None
    insights_model = None
    _insights_ok = False  # Flag: deps available, use per-operation connections
    if HAS_INSIGHTS:
        try:
            insights_conn = get_db()
            init_db(insights_conn)
            insights_conn.close()  # Don't hold a persistent connection
            insights_conn = None
            from fastembed import TextEmbedding
            insights_model = TextEmbedding("BAAI/bge-small-en-v1.5")
            _load_skip_list()
            if _insights_skip_set:
                logger.info("Insights skip list: %d sessions", len(_insights_skip_set))
            logger.info("Insights enabled (sqlite-vec + fastembed)")
            _DAEMON_STATUS["insights_enabled"] = True
            _insights_ok = True
        except Exception as e:
            logger.warning("Insights disabled: %s", e)
            insights_conn = None
            insights_model = None

    # Start LLM worker thread (single-threaded queue, no locks)
    _start_llm_worker(logger)

    _DAEMON_STATUS["state"] = "running"
    _write_status()

    # Start web child (claude-resume-duet) — managed subprocess
    _start_web_child(logger)

    # Start HUD progress bridge (Unix socket → HUD subprocess)
    hud_thread = threading.Thread(
        target=_start_hud_bridge, args=(shutdown, logger),
        name="hud-bridge", daemon=True,
    )
    hud_thread.start()

    logger.info(
        "Daemon started (pid=%d, poll=%ds, idle=%ds, llm=%s)",
        os.getpid(), POLL_INTERVAL_SECS, IDLE_THRESHOLD_SECS,
        "gemma-2b" if _local_llm_generate else "haiku",
    )

    while not shutdown.is_set():
        try:
            # Priority 1: Process on-demand task requests from TUI
            tasks_done = _process_task_files(caches, shutdown, logger)
            _DAEMON_STATUS["tasks_processed"] += tasks_done

            if shutdown.is_set():
                break

            # Priority 2: Background scan for idle uncached sessions
            sessions = _find_idle_uncached(caches, logger)
            _DAEMON_STATUS["queue_depth"] = len(sessions)
            _DAEMON_STATUS["last_run"] = datetime.now(timezone.utc).isoformat()
            _write_status()

            if sessions:
                logger.info("Found %d sessions to process", len(sessions))
                ok = 0
                fail = 0

                for s in sessions:
                    if shutdown.is_set():
                        logger.info("Shutdown requested, stopping mid-batch")
                        break

                    # Check for TUI tasks between background sessions
                    tasks_done = _process_task_files(caches, shutdown, logger)
                    _DAEMON_STATUS["tasks_processed"] += tasks_done

                    # Open a fresh DB connection per session to avoid
                    # holding write locks between operations
                    ic = None
                    if _insights_ok:
                        try:
                            ic = get_db()
                        except Exception:
                            pass

                    success = _analyze_session(
                        s, caches, logger,
                        insights_conn=ic,
                        insights_model=insights_model,
                    )

                    if ic is not None:
                        try:
                            ic.close()
                        except Exception:
                            pass

                    if success:
                        ok += 1
                        _DAEMON_STATUS["sessions_processed"] += 1
                        _DAEMON_STATUS["last_success"] = datetime.now(timezone.utc).isoformat()
                    else:
                        fail += 1
                        _DAEMON_STATUS["sessions_failed"] += 1
                        _DAEMON_STATUS["last_error"] = f"Session {s['session_id'][:8]}... failed"

                    _DAEMON_STATUS["queue_depth"] = max(0, _DAEMON_STATUS["queue_depth"] - 1)
                    _write_status()

                    if not shutdown.is_set():
                        time.sleep(RATE_LIMIT_SECS)

                logger.info("Batch complete: %d ok, %d failed", ok, fail)

            # Priority 3: Backfill insights for cached-but-unindexed sessions
            if _insights_ok and not shutdown.is_set():
                ic = None
                try:
                    ic = get_db()
                    backlog = _find_insights_backlog(
                        caches, ic, logger, batch_size=50,
                    )
                    ic.close()
                    ic = None
                except Exception:
                    if ic:
                        ic.close()
                    backlog = []

                if backlog:
                    logger.info("Insights backlog: %d sessions to index", len(backlog))
                    for s in backlog:
                        if shutdown.is_set():
                            break
                        tasks_done = _process_task_files(caches, shutdown, logger)
                        _DAEMON_STATUS["tasks_processed"] += tasks_done

                        ic = None
                        try:
                            ic = get_db()
                        except Exception:
                            pass
                        _index_session_only(
                            s, logger,
                            insights_conn=ic,
                            insights_model=insights_model,
                        )
                        if ic is not None:
                            try:
                                ic.close()
                            except Exception:
                                pass
                        if not shutdown.is_set():
                            time.sleep(RATE_LIMIT_SECS)

            # Update insights chunk count periodically
            if _insights_ok:
                try:
                    ic = get_db()
                    from .insights import get_stats
                    _DAEMON_STATUS["insights_chunks"] = get_stats(ic)["total_chunks"]
                    ic.close()
                except Exception:
                    pass

            # Priority 4: Scan for idle sessions missing window summaries
            # This is the backup plan — catches anything hooks missed
            if not shutdown.is_set():
                now = time.time()
                all_sessions = find_recent_sessions(LOOKBACK_HOURS, max_sessions=0)
                queued = 0
                for s in all_sessions:
                    if s["mtime"] > now - IDLE_THRESHOLD_SECS:
                        continue  # still active
                    sid = s["session_id"]
                    any_cached = False
                    for cache in caches:
                        ck = cache.cache_key(s["file"])
                        ws = cache.get(sid, ck, "window_summaries")
                        if ws and isinstance(ws, dict) and "5m" in ws:
                            any_cached = True
                            break
                    if not any_cached:
                        _compute_window_summaries(s, caches, logger)
                        queued += 1
                    if queued >= 10 or shutdown.is_set():
                        break  # cap per cycle — don't flood the queue
                if queued:
                    logger.info("Backfill: queued %d sessions for window summaries (queue: %d)",
                                queued, llm_queue_depth())

            # Priority 5: Generate L2 project summaries for stale projects
            if _insights_ok and HAS_HIERARCHY and not shutdown.is_set():
                try:
                    ic = get_db()
                    from .insights import list_stale_projects
                    stale = list_stale_projects(ic, limit=3)
                    ic.close()
                    ic = None

                    if stale:
                        logger.info("L2 summaries: %d stale projects", len(stale))
                        for proj in stale:
                            if shutdown.is_set():
                                break
                            ic = None
                            try:
                                ic = get_db()
                                result = generate_project_summary(
                                    proj["path"], ic, cache=caches[0],
                                )
                                if result:
                                    logger.info("L2 OK: %s -> %s",
                                                proj["name"], result.get("title", "?")[:40])
                                else:
                                    logger.debug("L2 SKIP: %s (no data)", proj["name"])
                            except Exception as he:
                                logger.warning("L2 FAIL: %s -> %s", proj["name"], he)
                            finally:
                                if ic:
                                    ic.close()
                            if not shutdown.is_set():
                                time.sleep(RATE_LIMIT_SECS)
                except Exception as he:
                    logger.warning("L2 cycle error: %s", he)
                    if ic:
                        try:
                            ic.close()
                        except Exception:
                            pass

            _DAEMON_STATUS["llm_queue_depth"] = llm_queue_depth()
            _DAEMON_STATUS["state"] = "idle"
            _write_status()

            # Ping web child to refresh its session list
            _notify_web_child()

        except Exception as e:
            logger.error("Error in poll cycle: %s", e, exc_info=True)
            _DAEMON_STATUS["last_error"] = str(e)
            _DAEMON_STATUS["state"] = "error"
            _write_status()

        # Interruptible sleep — but wake every 2s to check for TUI tasks
        elapsed = 0
        while elapsed < POLL_INTERVAL_SECS and not shutdown.is_set():
            shutdown.wait(TASK_POLL_SECS)
            elapsed += TASK_POLL_SECS
            if not shutdown.is_set():
                tasks_done = _process_task_files(caches, shutdown, logger)
                _DAEMON_STATUS["tasks_processed"] += tasks_done

        # Health-check web child at the top of each cycle
        _check_web_child(logger)
        _DAEMON_STATUS["state"] = "running"

    # Shutdown status
    _DAEMON_STATUS["state"] = "stopping"
    _write_status()

    # Stop web child before draining queue
    _stop_web_child(logger)

    # Drain LLM queue and stop worker
    _stop_llm_worker()
    logger.info("LLM worker stopped (queue drained)")

    _DAEMON_STATUS["state"] = "stopped"
    _write_status()
    logger.info("Daemon shutting down gracefully")


# ── Signal handling ──────────────────────────────────────────

def _setup_signals(shutdown: threading.Event, logger: logging.Logger):
    """Register SIGTERM and SIGINT handlers."""
    def _handler(signum, _frame):
        name = signal.Signals(signum).name
        logger.info("Received %s, initiating shutdown", name)
        shutdown.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# ── Entry point ──────────────────────────────────────────────

def main():
    """CLI entry point: claude-session-daemon."""
    if _check_already_running():
        print(f"Daemon already running (pid file: {PID_FILE})", file=sys.stderr)
        sys.exit(1)

    logger = _setup_logging()
    shutdown = threading.Event()

    _setup_signals(shutdown, logger)
    _write_pid()

    try:
        _run_loop(shutdown, logger)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down")
    finally:
        _remove_pid()
        logger.info("PID file removed, daemon stopped")
