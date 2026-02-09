"""Session classification — heuristic + ML ensemble.

Classifies Claude Code sessions as 'interactive' (human) or 'automated'
(programmatic). Uses a two-phase scan: count entry types/timestamps first,
then extract text features from user messages (filtering system prompts
and empty messages for clean signal).

The ML model (classifier.pkl) is loaded lazily on first use. If absent,
falls back to heuristic-only classification.
"""

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from .display import format_duration

# ── Dictionaries and markers ──────────────────────────────────

_POLITENESS_WORDS = {"please", "thanks", "thank you", "could you", "would you"}
_CASUAL_MARKERS = {
    "lol", "lmao", "lfg", "lgtm", "tbh", "imo", "fwiw", "nvm",
    "ok", "yep", "yea", "yeah", "nah", "hmm", "huh", "oh wait",
    "got it", "nice", "awesome", "cool", "btw", "brb", "idk",
    "omg", "wtf", "smh", "bet", "dope", "lit",
}

# System dictionary for misspelling detection (macOS)
_DICT_PATH = Path("/usr/share/dict/words")
_WORD_SET: set[str] = set()
if _DICT_PATH.exists():
    try:
        _WORD_SET = set(_DICT_PATH.read_text().lower().splitlines())
        _WORD_SET.update({
            "api", "cli", "ui", "ux", "json", "yaml", "toml", "csv", "sql",
            "html", "css", "js", "ts", "jsx", "tsx", "py", "rb", "rs", "go",
            "npm", "pip", "git", "github", "gitlab", "docker", "kubernetes",
            "async", "await", "const", "var", "func", "def", "impl", "enum",
            "str", "int", "bool", "dict", "tuple", "stdin", "stdout", "stderr",
            "url", "urls", "http", "https", "tcp", "udp", "ssh", "ssl", "tls",
            "env", "config", "configs", "args", "kwargs", "params", "middleware",
            "webhook", "webhooks", "endpoint", "endpoints", "repo", "repos",
            "readme", "changelog", "dockerfile", "makefile", "workflow",
            "codebase", "refactor", "refactored", "frontend", "backend",
            "fullstack", "devops", "ci", "cd", "pr", "prs", "todo", "todos",
            "auth", "oauth", "jwt", "cors", "csrf", "xss", "sdk", "mcp",
            "claude", "anthropic", "openai", "llm", "llms", "gpt", "ai",
        })
    except Exception:
        pass


# ── ML model (lazy load) ─────────────────────────────────────

_ML_MODEL = None
_ML_FEATURE_COLS = None


def _load_ml_model():
    """Load the serialized ML model (lazy, once)."""
    global _ML_MODEL, _ML_FEATURE_COLS
    if _ML_MODEL is not None:
        return True
    model_path = Path(__file__).parent / "classifier.pkl"
    if not model_path.exists():
        return False
    try:
        import pickle
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        _ML_MODEL = data["model"]
        _ML_FEATURE_COLS = data["feature_cols"]
        return True
    except Exception:
        return False


# ── Text extraction helpers ───────────────────────────────────


def _extract_user_text(obj: dict) -> str:
    """Extract user message text from a JSONL entry."""
    msg = obj.get("message", {}).get("content", "")
    if isinstance(msg, str):
        return msg.strip()
    if isinstance(msg, list):
        for c in msg:
            if isinstance(c, dict) and c.get("type") == "text":
                return c.get("text", "").strip()
    return ""


def _is_system_prompt(text: str) -> bool:
    """Detect if a user message is likely a programmatic system prompt."""
    if not text or len(text) < 200:
        return False
    lower = text[:300].lower()
    return (
        lower.startswith("you are ")
        or lower.startswith("research this ")
        or "\n# " in text[:500]
        or "instructions:" in lower
        or "your task is" in lower
        or "your role is" in lower
        or "your job is" in lower
    )


def _is_human_typo(word: str) -> bool:
    """Detect human transposition typos: swapping adjacent chars yields a real word."""
    if not _WORD_SET or len(word) < 3:
        return False
    for i in range(len(word) - 1):
        swapped = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if swapped in _WORD_SET:
            return True
    return False


# ── Scanning and feature extraction ──────────────────────────


def _count_entry(obj: dict, stats: dict):
    """Count types, timestamps, and assistant content. No user text features."""
    t = obj.get("type", "")
    ts = obj.get("timestamp")
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            stats["_timestamps"].append(dt.timestamp())
        except (ValueError, TypeError):
            pass

    if t == "user":
        stats["user_messages"] += 1
    elif t == "assistant":
        stats["assistant_messages"] += 1
        amsg = obj.get("message", {})
        if isinstance(amsg, dict):
            for c in (amsg.get("content") or []):
                if isinstance(c, dict):
                    if c.get("type") == "tool_use":
                        stats["tool_uses"] += 1
                    elif c.get("type") == "text":
                        txt = c.get("text", "")
                        stats["_asst_char_counts"].append(len(txt))
                        if "```" in txt:
                            stats["_asst_code_blocks"] += 1
    elif t == "tool_result":
        stats["tool_results"] += 1
    elif t == "system":
        stats["system_entries"] += 1
    elif t == "summary":
        stats["summary_entries"] += 1
    elif t == "progress":
        stats["progress_entries"] += 1


def _apply_user_text_features(text: str, stats: dict):
    """Extract text-based features from a single user message into stats."""
    words = text.split()
    stats["_user_word_counts"].append(len(words))
    stats["_user_char_counts"].append(len(text))
    lower = text.lower()
    if "?" in text:
        stats["_questions"] += 1
    if any(p in lower for p in _POLITENESS_WORDS):
        stats["_polite"] += 1
    if "```" in text:
        stats["_user_code_blocks"] += 1
    if any(c in lower for c in _CASUAL_MARKERS):
        stats["_casual"] += 1
    if text[0].islower():
        stats["_no_caps"] += 1
    if len(text) < 20:
        stats["_short_msgs"] += 1
    if "!" in text:
        stats["_exclamations"] += 1
    for w in words:
        clean = w.strip(".,!?;:'\"()-/").lower()
        if len(clean) >= 3 and clean.isalpha() and clean not in _WORD_SET:
            stats["_dict_words"] += 1
            if _is_human_typo(clean):
                stats["_misspelled"] += 1


def _new_scan_stats() -> dict:
    """Create a fresh stats accumulator for quick_scan."""
    return {
        "user_messages": 0, "assistant_messages": 0,
        "tool_uses": 0, "tool_results": 0,
        "system_entries": 0, "progress_entries": 0, "summary_entries": 0,
        "total_lines": 0,
        "_effective_users": 0, "_empty_messages": 0, "_first_is_prompt": False,
        "_timestamps": [], "_user_word_counts": [], "_user_char_counts": [],
        "_asst_char_counts": [], "_questions": 0, "_polite": 0,
        "_user_code_blocks": 0, "_asst_code_blocks": 0,
        "_casual": 0, "_no_caps": 0, "_short_msgs": 0,
        "_exclamations": 0, "_dict_words": 0, "_misspelled": 0,
    }


def _finalize_scan_stats(stats: dict, file_size: int) -> dict:
    """Convert raw scan accumulators into the feature dict used by classify + ML."""
    user = stats["user_messages"]
    effective = stats["_effective_users"]
    ts = stats["_timestamps"]
    duration = (max(ts) - min(ts)) if len(ts) >= 2 else 0

    secs_per_turn = (duration / effective) if effective > 0 else 0
    msgs_per_minute = (effective / (duration / 60)) if duration > 60 else 0
    tool_to_user = (stats["tool_uses"] / user) if user > 0 else 0

    q_ratio = (stats["_questions"] / effective) if effective > 0 else 0
    p_ratio = (stats["_polite"] / effective) if effective > 0 else 0
    avg_user_chars = float(np.mean(stats["_user_char_counts"])) if stats["_user_char_counts"] else 0
    avg_asst_chars = float(np.mean(stats["_asst_char_counts"])) if stats["_asst_char_counts"] else 0
    empty_ratio = (stats["_empty_messages"] / user) if user > 0 else 0

    return {
        "user_messages": user,
        "assistant_messages": stats["assistant_messages"],
        "tool_uses": stats["tool_uses"],
        "tool_results": stats["tool_results"],
        "system_entries": stats["system_entries"],
        "progress_entries": stats["progress_entries"],
        "summary_entries": stats["summary_entries"],
        "total_lines": stats["total_lines"],
        "file_size": file_size,
        "duration_secs": round(duration, 1),
        "log_duration": round(math.log1p(duration), 3),
        "secs_per_turn": round(secs_per_turn, 1),
        "msgs_per_minute": round(msgs_per_minute, 3),
        "tool_to_user_ratio": round(tool_to_user, 2),
        "question_ratio": round(q_ratio, 3),
        "politeness_ratio": round(p_ratio, 3),
        "avg_user_chars": round(avg_user_chars, 1),
        "user_code_blocks": stats["_user_code_blocks"],
        "avg_assistant_chars": round(avg_asst_chars, 1),
        "assistant_code_blocks": stats["_asst_code_blocks"],
        "casual_ratio": round((stats["_casual"] / effective) if effective > 0 else 0, 3),
        "no_caps_ratio": round((stats["_no_caps"] / effective) if effective > 0 else 0, 3),
        "short_msg_ratio": round((stats["_short_msgs"] / effective) if effective > 0 else 0, 3),
        "exclamation_ratio": round((stats["_exclamations"] / effective) if effective > 0 else 0, 3),
        "typo_score": round((stats["_misspelled"] / stats["_dict_words"]) if stats["_dict_words"] > 0 else 0, 3),
        "empty_msg_ratio": round(empty_ratio, 3),
        "first_is_prompt": 1 if stats["_first_is_prompt"] else 0,
        "has_progress": stats["progress_entries"] > 0,
        "duration_fmt": format_duration(duration),
        "classification": "pending",
    }


def quick_scan(session_file: Path) -> dict:
    """Scan a session file for classification features.

    Two-phase approach:
    1. Count types/timestamps/assistant features for every line
    2. Collect user texts, then apply text features after filtering
       system prompts and empty messages
    """
    stats = _new_scan_stats()
    size = 0
    user_texts = []

    try:
        size = session_file.stat().st_size
        with open(session_file) as fh:
            for line in fh:
                stats["total_lines"] += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _count_entry(obj, stats)
                if obj.get("type") == "user":
                    user_texts.append(_extract_user_text(obj))
    except Exception:
        pass

    texts_for_features = user_texts
    if texts_for_features and _is_system_prompt(texts_for_features[0]):
        stats["_first_is_prompt"] = True
        texts_for_features = texts_for_features[1:]

    for text in texts_for_features:
        if not text:
            stats["_empty_messages"] += 1
            continue
        stats["_effective_users"] += 1
        _apply_user_text_features(text, stats)

    if stats["_first_is_prompt"] and user_texts and not user_texts[0]:
        stats["_empty_messages"] += 1

    return _finalize_scan_stats(stats, size)


# ── Heuristic classifier ────────────────────────────────────


def classify_session(stats: dict) -> str:
    """Classify a session as 'interactive' or 'automated'.

    Primary signal: pace (seconds per user turn).
    Conservative: when in doubt, return 'interactive' (never hide real sessions).
    """
    user = stats.get("user_messages", 0)
    duration = stats.get("duration_secs", 0)
    has_progress = stats.get("has_progress", False)
    total_lines = stats.get("total_lines", 0)

    if total_lines <= 3:
        return "automated"

    if user <= 1 and stats.get("tool_uses", 0) == 0:
        return "automated"

    if has_progress:
        return "interactive"

    secs_per_turn = stats.get("secs_per_turn", 0)
    if user >= 2 and duration > 0 and secs_per_turn > 0:
        if secs_per_turn < 10:
            return "automated"
        if secs_per_turn > 30:
            return "interactive"

    if duration > 120 and user >= 3:
        return "interactive"

    if user >= 3:
        return "interactive"

    return "interactive"


# ── Ensemble classifier ──────────────────────────────────────


def _ensemble_classify(stats: dict) -> tuple[str, float]:
    """Heuristic + calibrated ML. Returns (label, confidence).

    Never calls Opus — that's handled by the public interface.
    """
    heuristic = classify_session(stats)

    if not _load_ml_model():
        return heuristic, 1.0

    try:
        features = np.array([[stats.get(col, 0) for col in _ML_FEATURE_COLS]])
        ml_pred = _ML_MODEL.predict(features)[0]
        ml_proba = _ML_MODEL.predict_proba(features)[0]
        ml_label = "interactive" if ml_pred == 1 else "automated"
        confidence = ml_proba.max()

        if heuristic == ml_label:
            return heuristic, confidence

        if confidence < 0.80:
            return "interactive", confidence

        if heuristic == "interactive" and ml_label == "automated":
            label = "automated" if confidence > 0.90 else "interactive"
            return label, confidence
        else:
            return "interactive", confidence
    except Exception:
        return heuristic, 1.0


# ── Opus fallback (for deep classification) ──────────────────


def _opus_classify(session_file: Path) -> str | None:
    """Use Opus via `claude -p` to classify a gray-zone session.

    Only called for sessions where the ML model is uncertain (conf < 0.80).
    Returns 'interactive' or 'automated', or None on failure.
    """
    user_msgs = []
    try:
        with open(session_file) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "user":
                        text = _extract_user_text(obj)
                        if text:
                            user_msgs.append(text[:200])
                except (json.JSONDecodeError, Exception):
                    continue
    except Exception:
        return None

    if not user_msgs:
        return None

    sample = user_msgs[:5]
    if len(user_msgs) > 5:
        sample += user_msgs[-5:]

    prompt = (
        "Classify this Claude Code session as 'interactive' (human typing) or 'automated' (programmatic/scripted).\n"
        "Human sessions have: typos, casual language, questions, short messages, varied tone.\n"
        "Automated sessions have: template prompts, system instructions, consistent formatting, role assignments.\n\n"
        "User messages (sample):\n"
    )
    for i, msg in enumerate(sample):
        prompt += f"[{i+1}] {msg}\n"
    prompt += "\nRespond with ONLY the word 'interactive' or 'automated'."

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "opus"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            answer = result.stdout.strip().lower()
            if "interactive" in answer:
                return "interactive"
            elif "automated" in answer:
                return "automated"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


# ── Public classification interface ──────────────────────────


def get_label(session_file: Path, cache: "SessionCache | None" = None) -> str:
    """Classify a session file -> 'interactive' or 'automated'.

    This is the only classification function the UI should call.
    Fast path only — no Opus calls. Use get_label_deep() for Opus fallback.
    """
    from .cache import SessionCache  # avoid circular import at module level

    sid = session_file.stem
    if cache:
        ck = cache.cache_key(session_file)
        cached = cache.get(sid, ck, "classification")
        if cached:
            return cached

    scan = quick_scan(session_file)
    label, _ = _ensemble_classify(scan)
    scan["classification"] = label

    if cache:
        ck = cache.cache_key(session_file)
        cache.set(sid, ck, "classification", label)
        cache.set(sid, ck, "stats", scan)

    return label


def get_label_deep(session_file: Path, cache: "SessionCache | None" = None) -> str:
    """Classify with Opus fallback for uncertain sessions.

    Use in batch mode (--cache-all), not in the interactive TUI.
    """
    from .cache import SessionCache

    sid = session_file.stem
    if cache:
        ck = cache.cache_key(session_file)
        cached = cache.get(sid, ck, "classification")
        if cached:
            return cached

    scan = quick_scan(session_file)
    label, confidence = _ensemble_classify(scan)

    if confidence < 0.80:
        opus_answer = _opus_classify(session_file)
        if opus_answer:
            label = opus_answer

    scan["classification"] = label

    if cache:
        ck = cache.cache_key(session_file)
        cache.set(sid, ck, "classification", label)
        cache.set(sid, ck, "stats", scan)

    return label
