#!/bin/bash
# Auto-train pipeline: waits for label gen to finish, then trains + evaluates.
set -e
cd /Users/dshanklinbv/repos-eidos-agi/claude-session-commons

TARGET_TRAIN=400   # minimum quality labels needed before training
CHECK_INTERVAL=60  # seconds between checks

echo "[pipeline] Waiting for label gen to finish (target: $TARGET_TRAIN train labels)..."

while true; do
    NLABELS=$(python3 -c "import json; d=json.load(open('training.labels.json')); print(len([v for v in d.values() if len(v)>=30]))" 2>/dev/null || echo 0)
    PROC=$(pgrep -fl "summarizer.dataset" | grep -c python || echo 0)
    echo "[pipeline] $(date '+%H:%M:%S') Train labels: $NLABELS | label gen processes: $PROC"

    # Stop waiting if: enough labels AND label gen finished
    if [ "$NLABELS" -ge "$TARGET_TRAIN" ] && [ "$PROC" -eq 0 ]; then
        echo "[pipeline] Ready. Starting dataset generation + training."
        break
    fi
    # Also break if label gen finished with fewer labels (take what we have)
    if [ "$PROC" -eq 0 ] && [ "$NLABELS" -gt 100 ]; then
        echo "[pipeline] Label gen finished with $NLABELS labels. Proceeding."
        break
    fi
    sleep $CHECK_INTERVAL
done

echo ""
echo "=== STEP 1: Generate dataset from labels ==="
python -m claude_session_commons.summarizer.dataset \
    --sessions train_filtered.txt --output training.jsonl \
    --generate-labels --n-sonnet 200 --n 1000 \
    --log label_gen_train.log
echo "training.jsonl rows: $(wc -l < training.jsonl)"

echo ""
echo "=== STEP 2: Generate test dataset ==="
python -m claude_session_commons.summarizer.dataset \
    --sessions test_filtered.txt --output test.jsonl \
    --generate-labels --n-sonnet 9999 --n 300 \
    --log label_gen_test.log
echo "test.jsonl rows: $(wc -l < test.jsonl)"

echo ""
echo "=== STEP 3: Fine-tune T5-small ==="
python -m claude_session_commons.summarizer.train \
    --dataset training.jsonl \
    --output ./my-summarizer \
    --epochs 4 \
    --batch-size 8 \
    2>&1 | tee train.log

echo ""
echo "=== STEP 4: Evaluate ==="
python -m claude_session_commons.summarizer.evaluate \
    --model ./my-summarizer \
    --test test.jsonl \
    --show 15 \
    2>&1 | tee evaluate.log

echo ""
echo "=== PIPELINE COMPLETE ==="
tail -10 evaluate.log
