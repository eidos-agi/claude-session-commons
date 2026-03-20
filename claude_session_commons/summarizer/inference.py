"""Step 4 of 4: Run inference with the quantized ONNX model.

Downloads the ONNX model on first use (to ~/.cache/claude-session-commons/summarizer/),
then runs inference locally with onnxruntime — no PyTorch, no network after first run.

This is the module that `_window_summary_adapter` in resume-resume calls.

Usage (standalone test):
    python -m claude_session_commons.summarizer.inference \
        --model-url https://huggingface.co/eidos-agi/session-summarizer/resolve/main \
        --text "User: let's fix the PyPI name collision..."

Public API:
    from claude_session_commons.summarizer.inference import summarize, is_available
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

# ── Model source ───────────────────────────────────────────────────────────────
# Override with CLAUDE_SUMMARIZER_URL env var to point at a custom model.
DEFAULT_MODEL_URL = os.environ.get(
    "CLAUDE_SUMMARIZER_URL",
    "https://huggingface.co/eidos-agi/session-summarizer/resolve/main",
)

CACHE_DIR = Path(os.environ.get(
    "CLAUDE_SUMMARIZER_CACHE",
    os.path.expanduser("~/.cache/claude-session-commons/summarizer"),
))

MODEL_FILES = [
    "encoder_model.onnx",
    "decoder_model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
]

# ── Singleton runtime ──────────────────────────────────────────────────────────
_enc_session = None
_dec_session = None
_tokenizer = None


def _download_if_needed(model_url: str = DEFAULT_MODEL_URL) -> bool:
    """Download model files to CACHE_DIR if not already present.

    Returns True if all files are available after the attempt.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in MODEL_FILES if not (CACHE_DIR / f).exists()]
    if not missing:
        return True

    print(f"[summarizer] Downloading model ({len(missing)} files) → {CACHE_DIR}")
    for fname in missing:
        url = f"{model_url.rstrip('/')}/{fname}"
        dest = CACHE_DIR / fname
        try:
            print(f"  {fname}…", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            kb = dest.stat().st_size // 1024
            print(f"{kb}KB")
        except Exception as e:
            print(f"FAILED: {e}")
            if dest.exists():
                dest.unlink()
            return False
    return True


def _load_sessions() -> bool:
    """Load ONNX sessions and tokenizer into module-level singletons.

    Returns True if successful.
    """
    global _enc_session, _dec_session, _tokenizer
    if _enc_session is not None:
        return True

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        _enc_session = ort.InferenceSession(str(CACHE_DIR / "encoder_model.onnx"))
        _dec_session = ort.InferenceSession(str(CACHE_DIR / "decoder_model.onnx"))
        _tokenizer = Tokenizer.from_file(str(CACHE_DIR / "tokenizer.json"))
        return True
    except Exception as e:
        print(f"[summarizer] Failed to load model: {e}")
        _enc_session = _dec_session = _tokenizer = None
        return False


def is_available(model_url: str = DEFAULT_MODEL_URL) -> bool:
    """Return True if the summarizer is ready to use.

    Downloads the model on first call if not cached. Subsequent calls
    are instant (checks module-level singleton).
    """
    if _enc_session is not None:
        return True
    if not _download_if_needed(model_url):
        return False
    return _load_sessions()


def summarize(
    text: str,
    max_new_tokens: int = 80,
    repetition_penalty: float = 1.3,
    model_url: str = DEFAULT_MODEL_URL,
) -> Optional[str]:
    """Summarize `text` using the local ONNX model.

    Returns a summary string, or None if the model is not available.

    Args:
        text: Raw window text (conversation + tool calls).
        max_new_tokens: Maximum output length in tokens (~75 tokens ≈ 50–60 words).
        repetition_penalty: Penalty for repeating tokens. 1.3 reduces looping.
        model_url: URL prefix to download model files from if not cached.
    """
    import numpy as np

    if not is_available(model_url):
        return None

    inp = "summarize: " + text.strip()
    enc_out = _tokenizer.encode(inp)
    ids = np.array([enc_out.ids], dtype=np.int64)
    mask = np.array([enc_out.attention_mask], dtype=np.int64)

    # Encode
    hidden = _enc_session.run(None, {"input_ids": ids, "attention_mask": mask})[0]

    # Greedy decode with repetition penalty
    decoder_ids = np.array([[0]], dtype=np.int64)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits = _dec_session.run(None, {
            "input_ids": decoder_ids,
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
        })[0]
        next_logits = logits[0, -1].copy()
        for prev in generated:
            if next_logits[prev] > 0:
                next_logits[prev] /= repetition_penalty
            else:
                next_logits[prev] *= repetition_penalty
        next_token = int(np.argmax(next_logits))
        if next_token == 1:  # EOS
            break
        generated.append(next_token)
        decoder_ids = np.concatenate([decoder_ids, [[next_token]]], axis=1)

    return _tokenizer.decode(generated, skip_special_tokens=True).strip() or None


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test ONNX summarizer inference")
    parser.add_argument("--text", help="Text to summarize (or omit for interactive mode)")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    if not is_available(args.model_url):
        print("Model not available — check model URL and internet connection.")
        return

    if args.text:
        result = summarize(args.text, max_new_tokens=args.max_tokens, model_url=args.model_url)
        print(result)
    else:
        print("Interactive mode (Ctrl-C to quit):")
        while True:
            try:
                text = input("\nText > ").strip()
                if text:
                    print("→", summarize(text, max_new_tokens=args.max_tokens, model_url=args.model_url))
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
