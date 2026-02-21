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
from .git_context import get_git_context
from .parse import parse_session
from .summarize import analyze_patterns, summarize_deep, summarize_quick

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
RESUME_CACHE_DIR = CLAUDE_DIR / "resume-summaries"
TASK_DIR = CLAUDE_DIR / "daemon-tasks"

IDLE_THRESHOLD_SECS = 600       # 10 minutes — don't process active sessions
POLL_INTERVAL_SECS = 300        # 5 minutes between scans
TASK_POLL_SECS = 2              # check for TUI task requests every 2s
RATE_LIMIT_SECS = 3             # pause between sessions (on top of claude -p time)
LOOKBACK_HOURS = 720            # 30 days — only process recent sessions
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
LOG_BACKUP_COUNT = 3


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


# ── Session filtering ────────────────────────────────────────

def _find_idle_uncached(
    caches: list[SessionCache],
    logger: logging.Logger,
) -> list[dict]:
    """Find sessions that are idle (10min+) and missing from any cache."""
    now = time.time()
    cutoff = now - IDLE_THRESHOLD_SECS
    all_sessions = find_recent_sessions(LOOKBACK_HOURS)

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
        if insights_conn is not None:
            try:
                turns, subagents = index_session(
                    str(session["file"]), insights_conn,
                    model=insights_model,
                    session_id=sid,
                    project_path=project_dir,
                )
                if turns or subagents:
                    logger.info("INSIGHTS: %s... -> %d turns, %d subagents",
                                sid[:8], turns, subagents)
            except Exception as ie:
                logger.warning("INSIGHTS FAIL: %s... -> %s", sid[:8], ie)

        title = summary.get("title", "?")
        logger.info("OK: %s... -> %s", sid[:8], title)
        return True

    except Exception as e:
        logger.warning("FAIL: %s... -> %s", sid[:8], e)
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

            count += 1

        except Exception as e:
            logger.warning("TASK FAIL [%s]: %s... -> %s", kind, sid[:8], e)

        tf.unlink(missing_ok=True)

    return count


# ── Main loop ────────────────────────────────────────────────

def _run_loop(shutdown: threading.Event, logger: logging.Logger):
    """Poll-and-process loop. Runs until shutdown event is set."""
    caches = [
        SessionCache(),
        SessionCache(RESUME_CACHE_DIR),
    ]

    # Initialize insights (optional — graceful if deps missing)
    insights_conn = None
    insights_model = None
    if HAS_INSIGHTS:
        try:
            insights_conn = get_db()
            init_db(insights_conn)
            from fastembed import TextEmbedding
            insights_model = TextEmbedding("BAAI/bge-small-en-v1.5")
            logger.info("Insights enabled (sqlite-vec + fastembed)")
        except Exception as e:
            logger.warning("Insights disabled: %s", e)
            insights_conn = None
            insights_model = None

    logger.info(
        "Daemon started (pid=%d, poll=%ds, idle=%ds)",
        os.getpid(), POLL_INTERVAL_SECS, IDLE_THRESHOLD_SECS,
    )

    while not shutdown.is_set():
        try:
            # Priority 1: Process on-demand task requests from TUI
            _process_task_files(caches, shutdown, logger)

            if shutdown.is_set():
                break

            # Priority 2: Background scan for idle uncached sessions
            sessions = _find_idle_uncached(caches, logger)

            if sessions:
                logger.info("Found %d sessions to process", len(sessions))
                ok = 0
                fail = 0

                for s in sessions:
                    if shutdown.is_set():
                        logger.info("Shutdown requested, stopping mid-batch")
                        break

                    # Check for TUI tasks between background sessions
                    _process_task_files(caches, shutdown, logger)

                    success = _analyze_session(
                        s, caches, logger,
                        insights_conn=insights_conn,
                        insights_model=insights_model,
                    )
                    if success:
                        ok += 1
                    else:
                        fail += 1

                    if not shutdown.is_set():
                        time.sleep(RATE_LIMIT_SECS)

                logger.info("Batch complete: %d ok, %d failed", ok, fail)

        except Exception as e:
            logger.error("Error in poll cycle: %s", e, exc_info=True)

        # Interruptible sleep — but wake every 2s to check for TUI tasks
        elapsed = 0
        while elapsed < POLL_INTERVAL_SECS and not shutdown.is_set():
            shutdown.wait(TASK_POLL_SECS)
            elapsed += TASK_POLL_SECS
            if not shutdown.is_set():
                _process_task_files(caches, shutdown, logger)

    if insights_conn is not None:
        try:
            insights_conn.close()
        except Exception:
            pass

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
