#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100nvl
#SBATCH --output=%x.%j.out
#SBATCH --partition=pradeepresearch
#SBATCH --qos=pradeepresearch-priority
#SBATCH --mail-user=sidney.tio.2021@phdcs.smu.edu.sg
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=jev-artist

# Serves jev (DiffusionGemma) with vLLM plus structured_server.py on one GPU,
# then runs the jev-artist over data/topics.yaml.
#
# Usage: sbatch scripts/artist.slurm [jev_ued.artist args]
#   sbatch scripts/artist.slurm                          # all topics, k=10
#   sbatch scripts/artist.slurm --only smiley,spiral --k 3
#
# Output: runs/artist-<jobid>/ (drawings, steps.jsonl) and its logs/ (servers).

MODEL="google/diffusiongemma-26B-A4B-it"
SERVED_NAME="dgemma"  # must match structured_server.py --model
CANVAS=64
ARTIST_ARGS=("$@")  # Everything after the script name goes to jev_ued.artist

module purge
module load Python/3.12.3-GCCcore-13.3.0  # check: module avail Python

cd ~/jev-ued
source .venv/bin/activate

export HF_HOME=~/scratch/huggingface_cache
# export HF_TOKEN=...  # only if the model download is refused

OUT_DIR="runs/artist-${SLURM_JOB_ID}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# The pinned vLLM wheel is the CUDA 13 build: it needs NVIDIA driver >= 580.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "GPU driver: $DRIVER"
if [[ "${DRIVER%%.*}" -lt 580 ]]; then
    echo "WARNING: driver < 580; install the cu129 vLLM wheel instead (see requirements.txt)"
fi

VLLM_PORT=$((8000 + ($RANDOM % 1000)))
JEV_PORT=$((9000 + ($RANDOM % 1000)))

VLLM_PID=""
JEV_PID=""
cleanup() {
    kill $JEV_PID $VLLM_PID 2>/dev/null
    wait $JEV_PID $VLLM_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

wait_for_server() {
    # First run downloads ~52GB of weights, so allow up to 60 x 30s.
    local port=$1 name=$2 pid=$3
    for i in $(seq 1 60); do
        curl -sf http://localhost:$port/health > /dev/null 2>&1 && echo "$name ready" && return 0
        if ! kill -0 $pid 2>/dev/null; then
            echo "ERROR: $name exited; see $LOG_DIR"
            return 1
        fi
        echo "Waiting for $name (attempt $i/60)..."
        sleep 30
    done
    echo "ERROR: $name failed to start"
    return 1
}

CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL \
    --served-model-name $SERVED_NAME \
    --port $VLLM_PORT \
    --diffusion-config "{\"canvas_length\":$CANVAS}" \
    --max-logprobs 32 \
    --enable-prefix-caching \
    > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

wait_for_server $VLLM_PORT "vllm" $VLLM_PID || exit 1

python jev_ued/structured_server.py \
    --upstream http://127.0.0.1:$VLLM_PORT \
    --model $SERVED_NAME \
    --tokenizer $MODEL \
    --canvas $CANVAS \
    --host 127.0.0.1 \
    --port $JEV_PORT \
    > "$LOG_DIR/structured_server.log" 2>&1 &
JEV_PID=$!

wait_for_server $JEV_PORT "structured_server" $JEV_PID || exit 1

python -m jev_ued.artist \
    --url http://127.0.0.1:$JEV_PORT \
    --out "$OUT_DIR" \
    "${ARTIST_ARGS[@]}"
STATUS=$?

echo "Artist exited with status $STATUS; results in $OUT_DIR"
exit $STATUS
