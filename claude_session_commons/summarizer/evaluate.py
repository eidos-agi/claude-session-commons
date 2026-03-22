"""Step 2.5 of 4: Evaluate a fine-tuned (or ONNX) summarizer on a held-out test set.

Computes ROUGE-1, ROUGE-2, ROUGE-L against Claude-generated reference labels.
Prints per-example output so you can eyeball quality alongside the aggregate scores.

Prerequisites:
    pip install rouge-score

Usage:
    # Evaluate a HuggingFace model directory
    python -m claude_session_commons.summarizer.evaluate \
        --model ./my-summarizer --test test.jsonl

    # Evaluate an ONNX model directory
    python -m claude_session_commons.summarizer.evaluate \
        --model ./my-summarizer-onnx --test test.jsonl --onnx

    # Show more per-example output
    python -m claude_session_commons.summarizer.evaluate \
        --model ./my-summarizer --test test.jsonl --show 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# -- Inference helpers ----------------------------------------------------------

def _predict_hf(model_dir: str, texts: list[str], batch_size: int = 8) -> list[str]:
    """Run inference with a HuggingFace model directory."""
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch

    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                repetition_penalty=1.3,
                num_beams=1,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


def _predict_onnx(model_dir: str, texts: list[str]) -> list[str]:
    """Run inference using the local ONNX model (no PyTorch needed)."""
    from claude_session_commons.summarizer import inference as inf_mod

    inf_mod._load_model(model_dir=model_dir)
    predictions = []
    for text in texts:
        pred = inf_mod._run_inference(text)
        predictions.append(pred or "")
    return predictions


# -- ROUGE scoring -------------------------------------------------------------

def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Return average ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("ERROR: rouge-score not installed. Run: pip install rouge-score", file=sys.stderr)
        sys.exit(1)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals: dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure

    n = max(len(predictions), 1)
    return {k: v / n for k, v in totals.items()}


# -- Main evaluation -----------------------------------------------------------

def evaluate(
    model_dir: str,
    test_jsonl: str,
    use_onnx: bool = False,
    show: int = 10,
    batch_size: int = 8,
) -> dict[str, float]:
    """Evaluate a model on a held-out test set.

    Prints per-example predictions (first `show` examples) and aggregate ROUGE scores.

    Args:
        model_dir: Path to HuggingFace model directory or ONNX model directory.
        test_jsonl: Path to test JSONL file (from dataset.py, with `summary` field).
        use_onnx: If True, use ONNX inference instead of HuggingFace.
        show: Number of per-example outputs to print.
        batch_size: Batch size for HuggingFace inference (ignored for ONNX).

    Returns:
        Dict with rouge1, rouge2, rougeL F1 scores.
    """
    rows = []
    with open(test_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print(f"No examples found in {test_jsonl}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating on {len(rows)} examples from {test_jsonl}")
    print(f"Model: {model_dir} ({'ONNX' if use_onnx else 'HuggingFace'})\n")

    texts = ["summarize: " + r["window_text"] for r in rows]
    references = [r["summary"] for r in rows]

    print("Running inference...")
    if use_onnx:
        predictions = _predict_onnx(model_dir, texts)
    else:
        predictions = _predict_hf(model_dir, texts, batch_size=batch_size)

    sep = "-" * 70
    print(f"\n{sep}")
    print(f"SAMPLE OUTPUT (first {min(show, len(rows))} examples)")
    print(sep)
    for i, (pred, ref, row) in enumerate(zip(predictions, references, rows)):
        if i >= show:
            break
        window_preview = row["window_text"][:120].replace("\n", " ")
        print(f"\n[{i+1}] Input:     {window_preview}...")
        print(f"     Reference: {ref}")
        print(f"     Predicted: {pred}")

    scores = compute_rouge(predictions, references)

    print(f"\n{sep}")
    print("ROUGE SCORES (F1, averaged over test set)")
    print(sep)
    print(f"  ROUGE-1: {scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {scores['rougeL']:.4f}")
    print()

    rouge_l = scores["rougeL"]
    if rouge_l < 0.25:
        print("=> ROUGE-L < 0.25: More data or longer training recommended (rerun steps 2-5)")
    elif rouge_l < 0.40:
        print("=> ROUGE-L 0.25-0.40: Acceptable -- ship and keep iterating")
    else:
        print("=> ROUGE-L > 0.40: Strong -- export to ONNX and ship")

    return scores


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a T5 summarizer on a held-out test set"
    )
    parser.add_argument("--model", required=True, help="HuggingFace or ONNX model directory")
    parser.add_argument("--test", required=True, help="Test JSONL file (from dataset.py)")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX inference")
    parser.add_argument(
        "--show", type=int, default=10,
        help="Number of per-example predictions to print (default: 10)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    evaluate(args.model, args.test, use_onnx=args.onnx, show=args.show, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
