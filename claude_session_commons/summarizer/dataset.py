"""Step 1 of 4: Generate a training dataset from Claude Code sessions.

Reads raw session JSONL files, extracts time-window slices (5m / 30m / 2h),
and pairs each window with a label produced by the best available summarizer.

Label strategy:
  - `--generate-labels`: call `claude -p` (Haiku for most, Sonnet for first N)
  - fallback: cached session_summary output, then last user message

Output: one JSONL file per line:
    {"window_text": "User: ...\nAssistant: ...", "summary": "...", "window": "30m"}

Usage:
    # Split sessions into train/test files
    python -m claude_session_commons.summarizer.dataset \\
        --split ~/.claude/projects --test-fraction 0.2

    # Generate labels for training set
    python -m claude_session_commons.summarizer.dataset \\
        --sessions train_split.txt --output training.jsonl \\
        --generate-labels --n-sonnet 200 --n 2000

    # Generate gold labels for test set (all Sonnet)
    python -m claude_session_commons.summarizer.dataset \\
        --sessions test_split.txt --output test.jsonl \\
        --generate-labels --n-sonnet 9999 --n 500
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator


# ── Window extraction ──────────────────────────────────────────────────────────

WINDOWS_MINUTES = {"5m": 5, "30m": 30, "2h": 120}


def _iter_messages(filepath: str) -> Iterator[dict]:
    """Yield parsed JSONL lines from a session file."""
    try:
        with open(filepath, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def extract_window(filepath: str, minutes: int) -> str:
    """Return the last `minutes` of conversation from a session file.

    Walks the JSONL tail, collecting user + assistant text until the
    time budget is exhausted (by timestamp) or we hit the start of file.
    Returns a plain-text block:

        User: <message>
        Assistant: <response>
        [Bash] <command>
        ...
    """
    lines: list[str] = []

    messages = list(_iter_messages(filepath))

    # Use the session's own last timestamp as the window anchor (not wall clock).
    # This ensures old sessions produce data, not just sessions from the last N minutes.
    def _parse_ts(ts) -> float:
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            try:
                from datetime import datetime
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return 0.0
        return 0.0

    last_ts = max((_parse_ts(m.get("timestamp", 0)) for m in messages), default=0.0)
    cutoff = (last_ts or __import__("time").time()) - minutes * 60

    for msg in messages:
        ts = _parse_ts(msg.get("timestamp", 0))
        if ts and ts < cutoff:
            continue

        # Top-level type field: "user" or "assistant" (schema v2)
        # Older sessions may have "role": "human"/"assistant" instead
        msg_type = msg.get("type") or msg.get("role", "")
        if msg_type in ("human",):
            msg_type = "user"

        # Content lives inside msg["message"]["content"] (schema v2)
        # or directly in msg["message"] (older schema)
        inner = msg.get("message", {})
        if isinstance(inner, dict):
            content = inner.get("content", inner)
        else:
            content = inner

        if msg_type == "user":
            parts = content if isinstance(content, list) else [content]
            for part in parts:
                if isinstance(part, str):
                    lines.append(f"User: {part.strip()}")
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        lines.append(f"User: {part['text'].strip()}")
                    elif part.get("type") == "tool_result":
                        snippet = str(part.get("content", ""))[:200]
                        lines.append(f"[Result] {snippet}")

        elif msg_type == "assistant":
            parts = content if isinstance(content, list) else [content]
            for part in parts:
                if isinstance(part, str):
                    lines.append(f"Assistant: {part.strip()}")
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        text = part["text"].strip()
                        if text:
                            lines.append(f"Assistant: {text[:500]}")
                    elif part.get("type") == "tool_use":
                        tool = part.get("name", "tool")
                        inp = part.get("input", {})
                        if "command" in inp:
                            lines.append(f"[{tool}] {inp['command'][:200]}")
                        elif "file_path" in inp:
                            lines.append(f"[{tool}] {inp['file_path']}")
                        else:
                            lines.append(f"[{tool}]")

    return "\n".join(lines).strip()


# ── Label generation ───────────────────────────────────────────────────────────

def _load_cached_summary(filepath: str) -> str | None:
    """Return a cached session summary if one exists alongside the session file.

    The cache lives at <session_id>.cache.json next to the JSONL, written by
    claude_session_commons.cache.SessionCache.
    """
    cache_path = filepath.replace(".jsonl", ".cache.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        # Cache stores summaries under arbitrary keys — find the first string value
        for v in data.values():
            if isinstance(v, dict):
                summary = v.get("summary", {})
                if isinstance(summary, dict):
                    return summary.get("what_was_done") or summary.get("state")
                if isinstance(summary, str) and len(summary) > 10:
                    return summary
        return None
    except (OSError, json.JSONDecodeError):
        return None


def _fallback_label(window_text: str) -> str:
    """Last-resort label: last user message, trimmed to 500 chars."""
    last = ""
    for line in window_text.splitlines():
        if line.startswith("User:"):
            last = line[5:].strip()
    text = last or window_text.strip()
    if len(text) > 500:
        text = text[:500].rsplit(" ", 1)[0] + "…"
    return text


def _get_origin(filepath: str) -> str:
    """Return 'human' or 'agent' for a session file.

    Uses classify.get_label() (fast, no LLM calls). Falls back to path
    heuristic if classify is unavailable.
    """
    try:
        from claude_session_commons.classify import get_label
        label = get_label(Path(filepath))
        return "human" if label == "interactive" else "agent"
    except Exception:
        pass
    # Path fallback: subagent sessions are always agent-spawned
    return "agent" if "/subagents/" in filepath else "human"


# ── Structured run logger ──────────────────────────────────────────────────────

class _RunLogger:
    """Append-only structured log for a dataset generation run."""

    def __init__(self, log_path: str | None):
        self._path = log_path
        self._start = time.time()
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Run started: {datetime.datetime.now().isoformat()}\n")
                f.write(f"{'='*70}\n")

    def log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self._path:
            with open(self._path, "a") as f:
                f.write(line + "\n")

    def log_label(self, i: int, total: int, tier: str, key: str,
                  success: bool, chars: int = 0, elapsed: float = 0.0) -> None:
        status = f"ok ({chars} chars, {elapsed:.1f}s)" if success else "failed"
        self.log(f"  [{i}/{total}] {tier}: {key[:55]}… {status}")

    def summary(self, generated: int, cached: int, failed: int, written: int,
                origin_counts: dict) -> None:
        elapsed = time.time() - self._start
        self.log(f"Labels: {generated} generated, {cached} cached, {failed} failed")
        self.log(f"Origin: {origin_counts.get('human', 0)} human, {origin_counts.get('agent', 0)} agent")
        self.log(f"Written: {written} examples | elapsed: {elapsed:.0f}s")


def _safe_env() -> dict:
    """Return os.environ with Claude Code subprocess vars stripped.

    These vars cause silent deadlock when calling claude -p from within
    a Claude Code session.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    return env


_SUMMARIZE_PROMPT = """\
You are a concise session summarizer. Read the conversation below and write \
2-3 sentences describing: (1) what the user was trying to accomplish, \
(2) what was actually completed, and (3) where the session stopped or what \
remains to be done. Be specific about file names, features, or bugs if \
mentioned. Output ONLY the summary sentences — no preamble, no bullet points.

Session:
{window_text}
"""


def _generate_label_via_claude(window_text: str, use_sonnet: bool = False) -> str | None:
    """Call claude -p to generate a summary label for a session window.

    Uses safe flags only: --no-session-persistence --output-format text --max-turns 1
    Strips CLAUDECODE + CLAUDE_CODE_ENTRYPOINT to prevent deadlock.

    Model is steered via prompt prefix (not --model flag, which causes hangs).
    """
    prompt = _SUMMARIZE_PROMPT.format(window_text=window_text[:3000])

    # Prepend a quality hint to steer toward higher-quality output for Sonnet-tier
    if use_sonnet:
        prompt = "Be thorough and precise.\n\n" + prompt

    try:
        # Resolve full path to claude binary — subprocess may not inherit shell PATH
        import shutil
        claude_bin = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")

        result = subprocess.run(
            [claude_bin, "-p", prompt,
             "--no-session-persistence",
             "--output-format", "text",
             "--max-turns", "1"],
            capture_output=True,
            text=True,
            timeout=60,
            env=_safe_env(),
        )
        if result.returncode != 0:
            return None
        summary = result.stdout.strip()
        return summary if len(summary) >= 30 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _labels_sidecar_path(output_jsonl: str) -> str:
    """Return path to the .labels.json sidecar for a given output file."""
    return output_jsonl.replace(".jsonl", ".labels.json")


def _load_label_cache(sidecar_path: str) -> dict[str, str]:
    """Load existing generated labels from sidecar (keyed by session basename)."""
    if not os.path.exists(sidecar_path):
        return {}
    try:
        with open(sidecar_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_label_cache(sidecar_path: str, cache: dict[str, str]) -> None:
    """Persist label cache to sidecar file."""
    with open(sidecar_path, "w") as f:
        json.dump(cache, f, indent=2)


def _label_one(args: tuple) -> tuple[str, str, str | None, float]:
    """Worker function for parallel label generation.

    Returns (key, tier, label_or_None, elapsed).
    """
    filepath, use_sonnet, minutes = args
    key = os.path.basename(filepath)
    tier = "sonnet" if use_sonnet else "haiku"
    t0 = time.time()
    window_text = extract_window(filepath, minutes)
    # 200-char floor: sparse sessions produce "No activity." noise labels
    if len(window_text) < 200:
        return key, tier, None, time.time() - t0
    # Skip circular sessions: resume-resume summarizer prompts masquerading as real work
    if "summarize what was happening in a claude code session" in window_text.lower():
        return key, tier, None, time.time() - t0
    label = _generate_label_via_claude(window_text, use_sonnet=use_sonnet)
    return key, tier, label, time.time() - t0


def generate_labels_for_sessions(
    filepaths: list[str],
    output_jsonl: str,
    window: str = "30m",
    n_sonnet: int = 200,
    log_path: str | None = None,
    workers: int = 6,
) -> dict[str, str]:
    """Generate Claude labels for a list of session files, saving to sidecar.

    Uses Sonnet-tier prompting for the first `n_sonnet` sessions (by index),
    Haiku-tier for the rest. Labels are cached in a .labels.json sidecar
    so re-runs are incremental. Runs `workers` parallel subprocess calls.

    Args:
        filepaths: Ordered list of session JSONL paths to label.
        output_jsonl: Path of the output dataset file (sidecar placed next to it).
        window: Time window for extraction ("5m", "30m", "2h").
        n_sonnet: Number of sessions (from the start of the list) to label with
                  Sonnet-tier quality prompting.
        log_path: Optional path to append structured log entries.
        workers: Number of parallel claude -p subprocess workers (default: 6).

    Returns:
        Dict mapping session basename → generated label.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    minutes = WINDOWS_MINUTES.get(window, 30)
    sidecar = _labels_sidecar_path(output_jsonl)
    cache = _load_label_cache(sidecar)
    logger = _RunLogger(log_path)

    # Pre-filter: skip already cached, build work list
    work = []
    skipped = 0
    for i, filepath in enumerate(filepaths):
        key = os.path.basename(filepath)
        if key in cache:
            skipped += 1
        else:
            work.append((filepath, i < n_sonnet, minutes))

    total_work = len(work)
    logger.log(f"Generating labels: {total_work} to process, {skipped} cached, "
               f"{workers} workers, n_sonnet={n_sonnet}, window={window}")

    generated = 0
    failed = 0
    done = 0
    save_every = max(1, workers * 2)  # persist to sidecar every N completions

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_label_one, args): args for args in work}
        for future in as_completed(futures):
            key, tier, label, elapsed = future.result()
            done += 1
            if label:
                cache[key] = label
                generated += 1
                logger.log_label(done, total_work, tier, key,
                                 success=True, chars=len(label), elapsed=elapsed)
            else:
                failed += 1
                logger.log_label(done, total_work, tier, key,
                                 success=False, elapsed=elapsed)
            # Persist periodically — resume-safe
            if done % save_every == 0:
                _save_label_cache(sidecar, cache)

    _save_label_cache(sidecar, cache)
    logger.log(f"Labels: {generated} generated, {skipped} cached, {failed} failed/sparse")
    return cache


# ── Session splitting ──────────────────────────────────────────────────────────

def split_sessions(
    sessions_dir: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    output_dir: str = ".",
) -> tuple[list[str], list[str]]:
    """Split session files into train and test sets at the session level.

    Splits BEFORE any label generation to prevent data leakage. Writes
    `train_split.txt` and `test_split.txt` in output_dir for reproducibility.

    Args:
        sessions_dir: Root directory containing Claude Code JSONL session files.
        test_fraction: Fraction of sessions to hold out for testing (default 0.2).
        seed: Random seed for reproducible splits.
        output_dir: Where to write the split text files.

    Returns:
        (train_files, test_files) — lists of absolute paths.
    """
    sessions = find_sessions(sessions_dir)
    if not sessions:
        print(f"No sessions found under {sessions_dir}", file=sys.stderr)
        return [], []

    random.seed(seed)
    shuffled = sessions[:]
    random.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_fraction))
    test_files = shuffled[:n_test]
    train_files = shuffled[n_test:]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_txt = out / "train_split.txt"
    test_txt = out / "test_split.txt"

    train_txt.write_text("\n".join(train_files) + "\n")
    test_txt.write_text("\n".join(test_files) + "\n")

    print(f"Split {len(sessions)} sessions → {len(train_files)} train / {len(test_files)} test")
    print(f"  train_split.txt → {train_txt}")
    print(f"  test_split.txt  → {test_txt}")
    return train_files, test_files


def _load_split_file(path: str) -> list[str]:
    """Read a split text file (one path per line) and return non-empty lines."""
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


# ── Dataset generation ─────────────────────────────────────────────────────────

def find_sessions(sessions_dir: str) -> list[str]:
    """Return all JSONL session files under sessions_dir."""
    root = Path(sessions_dir).expanduser()
    return [str(p) for p in root.rglob("*.jsonl") if p.stat().st_size > 1024]


def generate_dataset(
    sessions_dir: str = "~/.claude/projects",
    output: str = "training.jsonl",
    n: int = 2000,
    window: str = "30m",
    seed: int = 42,
    sessions_file: str | None = None,
    generate_labels: bool = False,
    n_sonnet: int = 200,
    log_path: str | None = None,
) -> int:
    """Generate a training dataset and write it to `output`.

    Returns the number of examples written.

    Args:
        sessions_dir: Root directory containing Claude Code JSONL session files.
                      Ignored when sessions_file is provided.
        output: Path to write the output JSONL.
        n: Maximum number of examples to generate.
        window: Time window to extract ("5m", "30m", or "2h").
        seed: Random seed for reproducible sampling (used when sessions_file is None).
        sessions_file: Path to a split text file (one session path per line).
                       When provided, uses this exact list instead of scanning sessions_dir.
        generate_labels: If True, call claude -p to generate labels instead of
                         using cached summaries.
        n_sonnet: When generate_labels=True, use Sonnet-tier prompting for the
                  first n_sonnet sessions (by list order).
    """
    minutes = WINDOWS_MINUTES.get(window, 30)

    if sessions_file:
        sessions = _load_split_file(sessions_file)
        # Filter to files that still exist and are large enough
        sessions = [s for s in sessions if os.path.exists(s) and os.path.getsize(s) > 1024]
    else:
        sessions = find_sessions(sessions_dir)
        random.seed(seed)
        random.shuffle(sessions)

    # Cap to n
    sessions = sessions[:n]

    logger = _RunLogger(log_path)
    logger.log(f"Dataset generation: output={output}, n={n}, window={window}, "
               f"generate_labels={generate_labels}, n_sonnet={n_sonnet}")
    logger.log(f"Sessions: {len(sessions)} candidates")

    # Pre-generate labels via claude -p if requested
    label_cache: dict[str, str] = {}
    if generate_labels:
        label_cache = generate_labels_for_sessions(
            sessions, output, window=window, n_sonnet=n_sonnet, log_path=log_path
        )

    written = 0
    origin_counts: dict[str, int] = {"human": 0, "agent": 0}
    label_sources: dict[str, int] = {"generated": 0, "cached": 0, "fallback": 0}

    with open(output, "w") as out_f:
        for filepath in sessions:
            if written >= n:
                break

            window_text = extract_window(filepath, minutes)
            if len(window_text) < 200:
                continue

            # Origin label — determines summary framing and T5 prefix
            origin = _get_origin(filepath)

            # Label priority: generated → cached → fallback
            key = os.path.basename(filepath)
            generated_label = label_cache.get(key)
            cached_label = _load_cached_summary(filepath)
            label = generated_label or cached_label or _fallback_label(window_text)

            if generated_label:
                label_sources["generated"] += 1
            elif cached_label:
                label_sources["cached"] += 1
            else:
                label_sources["fallback"] += 1

            if not label or len(label) < 10:
                continue

            # T5 input prefix conditioned on origin — teaches two distinct summary styles
            prefix = f"summarize {origin} session: "

            origin_counts[origin] = origin_counts.get(origin, 0) + 1
            out_f.write(json.dumps({
                "window_text": window_text[:2000],
                "input_text": prefix + window_text[:2000],  # conditioned T5 input
                "summary": label,
                "origin": origin,
                "window": window,
                "source": os.path.basename(os.path.dirname(filepath)),
            }) + "\n")
            written += 1

    logger.summary(
        generated=label_sources["generated"],
        cached=label_sources["cached"],
        failed=label_sources["fallback"],
        written=written,
        origin_counts=origin_counts,
    )
    logger.log(f"Label sources: generated={label_sources['generated']}, "
               f"cached={label_sources['cached']}, fallback={label_sources['fallback']}")
    return written


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summarization training data from Claude Code sessions"
    )

    # Split mode
    parser.add_argument(
        "--split",
        metavar="SESSIONS_DIR",
        help="Split sessions under this dir into train_split.txt / test_split.txt and exit",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Fraction of sessions to hold out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--split-output-dir", default=".",
        help="Where to write train_split.txt / test_split.txt (default: current dir)",
    )

    # Dataset generation
    parser.add_argument(
        "--sessions",
        default="~/.claude/projects",
        help="Root sessions directory OR path to a split .txt file",
    )
    parser.add_argument("--output", default="training.jsonl", help="Output JSONL file")
    parser.add_argument("--n", type=int, default=2000, help="Max examples")
    parser.add_argument("--window", default="30m", choices=["5m", "30m", "2h"])
    parser.add_argument("--seed", type=int, default=42)

    # Label generation
    parser.add_argument(
        "--generate-labels", action="store_true",
        help="Call claude -p to generate high-quality labels (instead of cached summaries)",
    )
    parser.add_argument(
        "--n-sonnet", type=int, default=200,
        help="Number of sessions to label with Sonnet-tier prompting (default: 200)",
    )
    parser.add_argument(
        "--log", metavar="LOG_FILE", default=None,
        help="Append structured run log to this file (e.g. label_gen_train.log)",
    )

    args = parser.parse_args()

    # Split mode: just split and exit
    if args.split:
        split_sessions(
            args.split,
            test_fraction=args.test_fraction,
            output_dir=args.split_output_dir,
        )
        return

    # Dataset generation mode
    # Detect if --sessions points to a .txt split file or a directory
    sessions_arg = args.sessions
    sessions_file = None
    sessions_dir = sessions_arg

    if sessions_arg.endswith(".txt") and os.path.isfile(sessions_arg):
        sessions_file = sessions_arg
        sessions_dir = "~/.claude/projects"  # unused when sessions_file is set

    generate_dataset(
        sessions_dir=sessions_dir,
        output=args.output,
        n=args.n,
        window=args.window,
        seed=args.seed,
        sessions_file=sessions_file,
        generate_labels=args.generate_labels,
        n_sonnet=args.n_sonnet,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
