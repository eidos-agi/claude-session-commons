#!/bin/bash
# Waits for training to finish, then evaluates
cd /Users/dshanklinbv/repos-eidos-agi/claude-session-commons
echo "[eval watcher] Waiting for training to finish (PID: $TRAIN_PID)..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
    STEP=$(grep -o '[0-9]*/[0-9]*' train.log | tail -1)
    echo "[eval watcher] $(date '+%H:%M:%S') progress: $STEP"
done
echo "[eval watcher] Training done. Running evaluation..."
python -m claude_session_commons.summarizer.evaluate \
    --model ./my-summarizer \
    --test test.jsonl \
    --show 20 \
    2>&1 | tee evaluate.log
echo "[eval watcher] DONE"
