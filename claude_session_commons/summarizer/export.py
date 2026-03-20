"""Step 3 of 4: Export a fine-tuned T5 model to quantized ONNX.

Converts the HuggingFace model directory produced by train.py into two
INT8-quantized ONNX files (encoder + decoder) that onnxruntime can run
without PyTorch.

Output layout:
    <output_dir>/
        encoder_model.onnx      (~35 MB quantized)
        decoder_model.onnx      (~41 MB quantized)
        tokenizer.json
        tokenizer_config.json
        spiece.model
        special_tokens_map.json

Usage:
    pip install optimum[onnxruntime] onnxruntime
    python -m claude_session_commons.summarizer.export \
        --model ./my-summarizer \
        --output ./my-summarizer-onnx
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path


def export(
    model_dir: str = "./my-summarizer",
    output_dir: str = "./my-summarizer-onnx",
) -> None:
    """Export a fine-tuned T5 model to quantized ONNX.

    Requires: optimum[onnxruntime], onnxruntime

    Args:
        model_dir: Fine-tuned HuggingFace model directory (from train.py).
        output_dir: Where to write ONNX files + tokenizer.
    """
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import tempfile

    os.makedirs(output_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="onnx_export_")

    print(f"Exporting {model_dir} → ONNX (fp32 intermediate)…")
    t0 = time.time()
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(model_dir, export=True)
    ort_model.save_pretrained(tmp)
    print(f"  exported in {time.time()-t0:.1f}s")

    # Quantize encoder + decoder
    for name in ("encoder_model.onnx", "decoder_model.onnx"):
        src = os.path.join(tmp, name)
        dst = os.path.join(output_dir, name)
        if not os.path.exists(src):
            print(f"  WARNING: {name} not found in export — skipping")
            continue
        orig_kb = os.path.getsize(src) // 1024
        print(f"  quantizing {name} ({orig_kb}KB)…")
        t0 = time.time()
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        q_kb = os.path.getsize(dst) // 1024
        print(f"    {orig_kb}KB → {q_kb}KB ({(1-q_kb/orig_kb)*100:.0f}% reduction) in {time.time()-t0:.1f}s")

    # Copy tokenizer files
    tokenizer_files = (
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "spiece.model",
    )
    for fname in tokenizer_files:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    shutil.rmtree(tmp, ignore_errors=True)

    total_mb = sum(
        os.path.getsize(os.path.join(output_dir, f)) // (1024 * 1024)
        for f in os.listdir(output_dir)
        if f.endswith(".onnx")
    )
    print(f"\nDone. ONNX model at {output_dir} (~{total_mb} MB total)")
    print("Next step: run inference.py to test, then upload to HuggingFace Hub.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export fine-tuned T5 to quantized ONNX")
    parser.add_argument("--model", default="./my-summarizer", help="Fine-tuned model dir")
    parser.add_argument("--output", default="./my-summarizer-onnx", help="ONNX output dir")
    args = parser.parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()
