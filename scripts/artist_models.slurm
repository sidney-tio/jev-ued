#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100nvl
#SBATCH --output=%x.%A_%a.out
#SBATCH --array=0-3
#SBATCH --partition=pradeepresearch
#SBATCH --qos=pradeepresearch-priority
#SBATCH --mail-user=sidney.tio.2021@phdcs.smu.edu.sg
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=jev-artist-models

# Slurm array over the kev models: each task serves one model with kev.serve
# (the kev repo's own /v1/systemone server; no vLLM) on one GPU, then
# runs the jev-artist over data/topics.yaml.
#
# kev needs torch<2.9, so it lives in its own venv, set up once:
#   git clone https://github.com/jaredpalmer/kev.git ~/kev
#   cd ~/kev && uv sync --extra serve
#
# Usage: sbatch scripts/artist_models.slurm [jev_ued.artist args]
#   sbatch scripts/artist_models.slurm                   # every model
#   sbatch --array=1,3 scripts/artist_models.slurm       # only kev-4b and kev-27b
#   sbatch scripts/artist_models.slurm --only smiley,spiral --k 3
#
# Output: runs/artist-<model>-<arrayjobid>/ (drawings, steps.jsonl)
# and its logs/ (server), e.g. runs/artist-kev-4b-123456/. Slurm logs go to
# jev-artist-models.<arrayjobid>_<taskid>.out.

# Index = SLURM_ARRAY_TASK_ID; keep --array above in sync with this list.
MODELS=(
    "jaredpalmer/kev-0.8b"
    "jaredpalmer/kev-4b"
    "jaredpalmer/kev-9b"
    "jaredpalmer/kev-27b"
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "$MODEL" ]]; then
    echo "ERROR: no model for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi
MODEL_TAG="${MODEL##*/}"  # e.g. kev-4b, used in the output dir name
KEV_DIR=~/kev  # kev checkout with its own .venv (see above)
ARTIST_ARGS=("$@")  # Everything after the script name goes to jev_ued.artist

module purge
module load Python/3.12.3-GCCcore-13.3.0  # check: module avail Python

cd ~/jev-ued
source .venv/bin/activate

export HF_HOME=~/scratch/huggingface_cache
# export HF_TOKEN=...  # only if the model download is refused

OUT_DIR="runs/artist-${MODEL_TAG}-${SLURM_ARRAY_JOB_ID}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "Model: $MODEL -> $OUT_DIR"

if [[ ! -x "$KEV_DIR/.venv/bin/python" ]]; then
    echo "ERROR: no kev venv at $KEV_DIR/.venv; see the setup above"
    exit 1
fi

KEV_PORT=$((9000 + ($RANDOM % 1000)))

KEV_PID=""
cleanup() {
    kill $KEV_PID 2>/dev/null
    wait $KEV_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

wait_for_server() {
    # kev.serve has no /health; /v1/models answers once the model is loaded.
    # First run downloads the weights (~54GB for kev-27b), so allow up to 60 x 30s.
    local port=$1 name=$2 pid=$3
    for i in $(seq 1 60); do
        curl -sf http://localhost:$port/v1/models > /dev/null 2>&1 && echo "$name ready" && return 0
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

CUDA_VISIBLE_DEVICES=0 "$KEV_DIR/.venv/bin/python" -m kev.serve \
    --run $MODEL \
    --host 127.0.0.1 \
    --port $KEV_PORT \
    > "$LOG_DIR/kev_serve.log" 2>&1 &
KEV_PID=$!

wait_for_server $KEV_PORT "kev.serve" $KEV_PID || exit 1

# kev.serve drops a top-level instructions field, hence --instructions-in-state.
python -m jev_ued.artist \
    --url http://127.0.0.1:$KEV_PORT \
    --out "$OUT_DIR" \
    --instructions-in-state \
    "${ARTIST_ARGS[@]}"
STATUS=$?

echo "Artist ($MODEL) exited with status $STATUS; results in $OUT_DIR"
exit $STATUS
