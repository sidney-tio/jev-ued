#!/usr/bin/env bash
# Serves jev (DiffusionGemma) with vLLM, then the structured-read server in
# front of it. Clients talk to http://127.0.0.1:8011 (/v1/systemone).
# Logs go to logs/. Stop both with: kill $(cat logs/*.pid)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=google/diffusiongemma-26B-A4B-it
SERVED_NAME=dgemma  # must match structured_server.py --model
CANVAS=64
VLLM_PORT=8000
PORT=8011

mkdir -p logs

vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$VLLM_PORT" \
  --diffusion-config "{\"canvas_length\":$CANVAS}" \
  --max-logprobs 32 \
  --enable-prefix-caching \
  > logs/vllm.log 2>&1 &
echo $! > logs/vllm.pid

echo "Waiting for vLLM on :$VLLM_PORT (model load can take several minutes)..."
until curl -fsS "localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
  if ! kill -0 "$(cat logs/vllm.pid)" 2>/dev/null; then
    echo "vLLM exited; see logs/vllm.log" >&2
    exit 1
  fi
  sleep 5
done

python jev_ued/structured_server.py \
  --upstream "http://127.0.0.1:$VLLM_PORT" \
  --model "$SERVED_NAME" \
  --tokenizer "$MODEL" \
  --canvas "$CANVAS" \
  --port "$PORT" \
  > logs/structured_server.log 2>&1 &
echo $! > logs/structured_server.pid

until curl -fsS "localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done
echo "Ready: http://127.0.0.1:$PORT/v1/systemone"
