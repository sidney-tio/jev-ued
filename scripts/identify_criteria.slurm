#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100nvl
#SBATCH --output=%x.%A_%a.out
#SBATCH --array=0-11%4  # 4 models x 3 criteria; at most 4 tasks run at once
#SBATCH --partition=pradeepresearch
#SBATCH --qos=pradeepresearch-priority
#SBATCH --mail-user=sidney.tio.2021@phdcs.smu.edu.sg
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=jev-identify-criteria

# Slurm array over the kev models x the choice criteria: each task serves one
# model with kev.serve (the kev repo's own /v1/systemone server; no vLLM) on
# one GPU, then runs jev-identify --mode choice with one --criteria over the
# target pictures in data/topics.yaml. %4 on --array keeps at most 4 tasks
# (4 GPUs) running at once; the rest wait in the queue.
#
# kev needs torch<2.9, so it lives in its own venv, set up once:
#   git clone https://github.com/jaredpalmer/kev.git ~/kev
#   cd ~/kev && uv sync --extra serve
#
# Usage: sbatch scripts/identify_criteria.slurm [jev_ued.identify args]
#   sbatch scripts/identify_criteria.slurm                 # all 12 runs, k=5
#   sbatch scripts/identify_criteria.slurm --k 3
#   sbatch --array=9-11 scripts/identify_criteria.slurm    # only kev-27b
# --mode and --criteria come from the array index; passing them is an error.
#
# Output: runs/identify-choice-<criteria>-<model>-<arrayjobid>/
# (answers.jsonl, summary.json) and its logs/ (server), e.g.
# runs/identify-choice-null-kev-4b-123456/. Slurm logs go to
# jev-identify-criteria.<arrayjobid>_<taskid>.out.

# Task i runs MODELS[i / 3] with CRITERIA[i % 3], so one model's runs are
# adjacent: 0-2 kev-0.8b, 3-5 kev-4b, 6-8 kev-9b, 9-11 kev-27b. Keep --array
# above at 0-(models x criteria - 1).
MODELS=(
    "jaredpalmer/kev-0.8b"
    "jaredpalmer/kev-4b"
    "jaredpalmer/kev-9b"
    "jaredpalmer/kev-27b"
)
CRITERIA_LIST=(null name probe)
N_CRITERIA=${#CRITERIA_LIST[@]}
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID / N_CRITERIA))]}"
CRITERIA="${CRITERIA_LIST[$((SLURM_ARRAY_TASK_ID % N_CRITERIA))]}"
if [[ -z "$MODEL" ]]; then
    echo "ERROR: no model for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi
MODEL_TAG="${MODEL##*/}"  # e.g. kev-4b, used in the output dir name
KEV_DIR=~/kev  # kev checkout with its own .venv (see above)
IDENTIFY_ARGS=("$@")  # Everything after the script name goes to jev_ued.identify

module purge
module load Python/3.12.3-GCCcore-13.3.0  # check: module avail Python

cd ~/jev-ued
source .venv/bin/activate

export HF_HOME=~/scratch/huggingface_cache
# export HF_TOKEN=...  # only if the model download is refused

# The array index sets the mode and criteria; don't let the args contradict it
for arg in "${IDENTIFY_ARGS[@]}"; do
    case "$arg" in
        --mode|--mode=*|--criteria|--criteria=*)
            echo "ERROR: $arg: this script sets --mode choice and --criteria per task"
            exit 1 ;;
    esac
done
OUT_DIR="runs/identify-choice-${CRITERIA}-${MODEL_TAG}-${SLURM_ARRAY_JOB_ID}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "Model: $MODEL, criteria: $CRITERIA -> $OUT_DIR"

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
python -m jev_ued.identify \
    --url http://127.0.0.1:$KEV_PORT \
    --out "$OUT_DIR" \
    --instructions-in-state \
    --mode choice \
    --criteria "$CRITERIA" \
    "${IDENTIFY_ARGS[@]}"
STATUS=$?

echo "Identify ($MODEL, criteria $CRITERIA) exited with status $STATUS; results in $OUT_DIR"
exit $STATUS
