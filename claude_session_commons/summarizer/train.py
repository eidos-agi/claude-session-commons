"""Step 2 of 4: Fine-tune T5-small on window → summary pairs.

Reads the JSONL dataset produced by dataset.py and fine-tunes a T5-small
checkpoint. Runs on CPU or Apple Silicon (via MPS/MLX); no GPU required,
though it's faster with one.

Produces a standard HuggingFace model directory that export.py will convert
to ONNX.

Usage:
    pip install transformers datasets torch
    python -m claude_session_commons.summarizer.train \
        --dataset training.jsonl \
        --output ./my-summarizer \
        --epochs 3 \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_hf_dataset(rows: list[dict]):
    """Convert list of dicts to a HuggingFace Dataset with train/eval split."""
    from datasets import Dataset

    # Use pre-built input_text (origin-conditioned) when present in dataset rows.
    # Falls back to generic "summarize: " prefix for older datasets.
    dataset = Dataset.from_list([
        {
            "input_text": r.get("input_text") or ("summarize: " + r["window_text"]),
            "target_text": r["summary"],
            "origin": r.get("origin", "unknown"),
        }
        for r in rows
    ])
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # Log origin distribution
    for split_name, ds in [("train", split["train"]), ("eval", split["test"])]:
        origins = {}
        for o in ds["origin"]:
            origins[o] = origins.get(o, 0) + 1
        print(f"  {split_name}: {dict(origins)}")

    return split["train"], split["test"]


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    dataset: str = "training.jsonl",
    output_dir: str = "./my-summarizer",
    base_model: str = "t5-small",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    max_input_len: int = 512,
    max_target_len: int = 64,
) -> None:
    """Fine-tune T5-small on the dataset and save to output_dir.

    Args:
        dataset: Path to JSONL training file (from dataset.py).
        output_dir: Where to write the fine-tuned model.
        base_model: HuggingFace model ID to start from (default: t5-small).
        epochs: Training epochs. 3 is usually enough for small datasets.
        batch_size: Per-device batch size. Lower if you hit OOM.
        learning_rate: AdamW learning rate. 5e-4 works well for T5-small.
        max_input_len: Max tokens for window text. 512 covers most windows.
        max_target_len: Max tokens for summary output. 64 = ~50 words.
    """
    from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EvalPrediction,  # noqa: F401 — used in compute_metrics type hint
    )
    import numpy as np  # noqa: F401 — used in compute_metrics
    import torch

    print(f"Loading base model: {base_model}")
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    model = T5ForConditionalGeneration.from_pretrained(base_model)

    print(f"Loading dataset: {dataset}")
    rows = load_jsonl(dataset)
    print(f"  {len(rows)} examples")
    train_ds, eval_ds = build_hf_dataset(rows)

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_len,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length=max_target_len,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])

    # Detect device
    if torch.backends.mps.is_available():
        device_note = "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        device_note = "CUDA GPU"
    else:
        device_note = "CPU (slow — consider reducing epochs or dataset size)"
    print(f"Training on: {device_note}")

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),   # fp16 only on CUDA
        push_to_hub=False,
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    def compute_metrics(pred: EvalPrediction) -> dict:
        """Compute ROUGE scores after each eval epoch."""
        try:
            from rouge_score import rouge_scorer as rs_mod
        except ImportError:
            return {}

        scorer = rs_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        labels = pred.label_ids
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions

        # Replace -100 (padding) with pad token id before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        totals: dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for p, r in zip(decoded_preds, decoded_labels):
            scores = scorer.score(r, p)
            for key in totals:
                totals[key] += scores[key].fmeasure

        n = max(len(decoded_preds), 1)
        results = {k: round(v / n, 4) for k, v in totals.items()}
        print(f"\n  ROUGE: {results}")
        return results

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune T5-small on Claude Code session windows")
    parser.add_argument("--dataset", default="training.jsonl")
    parser.add_argument("--output", default="./my-summarizer")
    parser.add_argument("--base-model", default="t5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    train(args.dataset, args.output, args.base_model, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
