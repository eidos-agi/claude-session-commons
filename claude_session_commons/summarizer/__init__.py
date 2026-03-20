"""Session summarizer factory — end-to-end pipeline.

Four steps to build a domain-specific session summarizer:

    Step 1 — Generate training data:
        python -m claude_session_commons.summarizer.dataset \\
            --sessions ~/.claude/projects --output training.jsonl --n 2000

    Step 2 — Fine-tune T5-small:
        pip install transformers datasets torch
        python -m claude_session_commons.summarizer.train \\
            --dataset training.jsonl --output ./my-summarizer

    Step 3 — Export to quantized ONNX:
        pip install optimum[onnxruntime]
        python -m claude_session_commons.summarizer.export \\
            --model ./my-summarizer --output ./my-summarizer-onnx

    Step 4 — Run inference (no PyTorch required):
        pip install onnxruntime tokenizers
        from claude_session_commons.summarizer import summarize, is_available

        if is_available():
            print(summarize("User: let's rename the package..."))

The inference module downloads the model on first call and caches it to
~/.cache/claude-session-commons/summarizer/. Point CLAUDE_SUMMARIZER_URL
at your own HuggingFace repo to use a custom fine-tuned checkpoint.
"""

from .inference import is_available, summarize

__all__ = ["is_available", "summarize"]
