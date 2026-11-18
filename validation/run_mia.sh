#!/bin/bash

set -uo pipefail

EVAL_SIZE=${1:-25000}
NUM_SHADOWS=${2:-64}
CW_BATCH_SIZE=${3:-512}
NORM_LAYER=${4:-lnchw}

GPUS=(0)
NUM_GPUS=${#GPUS[@]}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-1}
DATASETS=("cifar10" "cifar100")
MODELS=("resnet50" "resnet34" "resnet18")
SEEDS=(42 43 44)
EPOCHS=${EPOCHS:-200}
CHUNK_SIZE=${CHUNK_SIZE:-4}
EXTRA_ARGS=${EXTRA_ARGS:-}
PYTHON=${PYTHON:-python}

OUT_DIR="./results"
PLOT_DIR="./plots"
DATA_DIR="./data"
LOG_DIR="${OUT_DIR}/logs_mia"

TOTAL_SLOTS=$(( NUM_GPUS * SLOTS_PER_GPU ))
CPUS=$(nproc 2>/dev/null || echo 8)
NUM_WORKERS=${NUM_WORKERS:-$(( CPUS / TOTAL_SLOTS ))}
if [ "$NUM_WORKERS" -lt 2 ]; then NUM_WORKERS=2; fi
if [ "$NUM_WORKERS" -gt 8 ]; then NUM_WORKERS=8; fi

mkdir -p "$OUT_DIR" "$PLOT_DIR" "$DATA_DIR" "$LOG_DIR"
export PYTHONUNBUFFERED=1

echo "=================================================================="
echo " Starting ${NUM_GPUS}-GPU Parallel MIA Half-Dataset (25k) Pipeline"
echo " GPUs: ${GPUS[*]} x ${SLOTS_PER_GPU} slot(s) | C&W Eval Size: ${EVAL_SIZE} | Shadows: ${NUM_SHADOWS} | Norm: ${NORM_LAYER}"
echo " Epochs: ${EPOCHS} | Shadow chunk: ${CHUNK_SIZE} | DataLoader workers/job: ${NUM_WORKERS} | Extra: ${EXTRA_ARGS}"
echo "=================================================================="

# Download datasets once before any parallel job (avoids download races).
"$PYTHON" -c "
from torchvision import datasets
for cls in (datasets.CIFAR10, datasets.CIFAR100):
    cls(root='${DATA_DIR}', train=True, download=True)
    cls(root='${DATA_DIR}', train=False, download=True)
" || { echo "[!] Dataset download failed"; exit 1; }

# Create all splits sequentially before parallel jobs (avoids split-file races).
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        "$PYTHON" train_eval_mia.py --dataset "$DATASET" --model "$MODEL" --norm-layer "$NORM_LAYER" \
            --phase init-splits --num-shadows "$NUM_SHADOWS" --eval-size "$EVAL_SIZE" \
            --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" || { echo "[!] init-splits failed"; exit 1; }
    done
done

# Job list, longest first: per model (R50 -> R34 -> R18): targets (training + C&W), then shadow chunks.
JOBS=()
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            JOBS+=("TARGET:${DATASET}:${MODEL}:${SEED}:0")
        done
    done
    for DATASET in "${DATASETS[@]}"; do
        for ((s = 0; s < NUM_SHADOWS; s += CHUNK_SIZE)); do
            s_end=$(( s + CHUNK_SIZE ))
            if [ "$s_end" -gt "$NUM_SHADOWS" ]; then s_end=$NUM_SHADOWS; fi
            JOBS+=("SHADOW:${DATASET}:${MODEL}:${s}:${s_end}")
        done
    done
done
TOTAL_JOBS=${#JOBS[@]}
echo "[*] Dispatching ${TOTAL_JOBS} MIA jobs (target seeds + shadow chunks) across ${TOTAL_SLOTS} GPU slot(s)..."

QUEUE_DIR=$(mktemp -d)
COUNTER_FILE="${QUEUE_DIR}/next"
LOCK_DIR="${QUEUE_DIR}/lock"
FAILED_FILE="${LOG_DIR}/failed_jobs.txt"
echo 0 > "$COUNTER_FILE"
: > "$FAILED_FILE"
cleanup() { rm -rf "$QUEUE_DIR"; }
trap cleanup EXIT
trap 'echo "[!] Interrupted - stopping all jobs"; kill 0' INT TERM

next_job() {   # atomic fetch-and-increment (mkdir is an atomic lock)
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do sleep 0.05; done
    local i
    i=$(cat "$COUNTER_FILE")
    echo $(( i + 1 )) > "$COUNTER_FILE"
    rmdir "$LOCK_DIR"
    echo "$i"
}

COMMON_ARGS=(--norm-layer "$NORM_LAYER" --num-shadows "$NUM_SHADOWS" --eval-size "$EVAL_SIZE"
             --epochs "$EPOCHS" --batch-size 128 --cw-batch-size "$CW_BATCH_SIZE" --lr 0.1
             --num-workers "$NUM_WORKERS" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR")

worker() {
    local gpu=$1 slot=$2
    local idx jtype dataset model a1 a2 log desc
    while true; do
        idx=$(next_job)
        if [ "$idx" -ge "$TOTAL_JOBS" ]; then break; fi
        IFS=":" read -r jtype dataset model a1 a2 <<< "${JOBS[$idx]}"
        if [ "$jtype" == "TARGET" ]; then
            log="${LOG_DIR}/mia_target_${dataset}_${model}_seed${a1}_gpu${gpu}.log"
            desc="Target ${dataset} | ${model} | seed ${a1}"
            JOB_ARGS=(--phase target --seeds "$a1")
        else
            log="${LOG_DIR}/mia_shadow_${dataset}_${model}_${a1}_to_${a2}_gpu${gpu}.log"
            desc="Shadows $(( a1 + 1 ))..${a2} | ${dataset} | ${model}"
            JOB_ARGS=(--phase shadow --shadow-start "$a1" --shadow-end "$a2")
        fi
        echo "[$(date '+%H:%M:%S')] [Job $(( idx + 1 ))/${TOTAL_JOBS}] GPU ${gpu}.${slot}: ${desc} -> ${log}"
        # shellcheck disable=SC2086
        if CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" train_eval_mia.py --dataset "$dataset" --model "$model" \
                "${JOB_ARGS[@]}" "${COMMON_ARGS[@]}" $EXTRA_ARGS > "$log" 2>&1; then
            echo "[$(date '+%H:%M:%S')] [Job $(( idx + 1 ))/${TOTAL_JOBS}] DONE ${desc}"
        else
            echo "[$(date '+%H:%M:%S')] [!] FAILED ${desc} (see ${log})"
            echo "${JOBS[$idx]}" >> "$FAILED_FILE"
        fi
    done
}

PIDS=()
for ((g = 0; g < NUM_GPUS; g++)); do
    for ((s = 0; s < SLOTS_PER_GPU; s++)); do
        worker "${GPUS[$g]}" "$s" &
        PIDS+=($!)
    done
done
echo "[*] Waiting for all GPU workers to finish..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

N_FAILED=$(wc -l < "$FAILED_FILE" | tr -d ' ')
if [ "$N_FAILED" -gt 0 ]; then
    echo "[!] ${N_FAILED} job(s) failed:"
    sed 's/^/    /' "$FAILED_FILE"
    echo "[!] Re-run the same command: finished targets/shadows are detected and skipped automatically."
else
    echo "[+] All target and shadow jobs finished!"
fi

echo "=================================================================="
echo " Aggregating MIA Metrics & Generating 8-Panel Verification Plots"
echo "=================================================================="
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "resnet18" "resnet34" "resnet50"; do
        if ! "$PYTHON" train_eval_mia.py --dataset "$DATASET" --model "$MODEL" --norm-layer "$NORM_LAYER" \
                --phase aggregate --target-seeds-for-merge "${SEEDS[@]}" --num-shadows "$NUM_SHADOWS" \
                --eval-size "$EVAL_SIZE" --epochs "$EPOCHS" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"; then
            echo "[!] Aggregation failed for ${DATASET}/${MODEL} (missing targets/shadows?) - skipping plots."
            continue
        fi
        FULL_CSV="${OUT_DIR}/full_${DATASET}_${MODEL}_3seeds.csv"
        MIA_CSV="${OUT_DIR}/mia_${DATASET}_${MODEL}_3seeds.csv"
        if [ ! -f "$MIA_CSV" ]; then
            echo "[!] ${MIA_CSV} missing - skipping plots."
            continue
        fi
        if [ -f "$FULL_CSV" ]; then
            TOP_CSV="$FULL_CSV"
        else
            echo "[*] ${FULL_CSV} not found - using the MIA CSV for the top row as well."
            TOP_CSV="$MIA_CSV"
        fi
        "$PYTHON" plot_verification.py --full-csv "$TOP_CSV" --mia-csv "$MIA_CSV" \
            --dataset "$DATASET" --model "$MODEL" --out-dir "$PLOT_DIR" || echo "[!] Plotting failed for ${DATASET}/${MODEL}"
    done
done

if [ "$N_FAILED" -gt 0 ]; then
    echo "[!] MIA pipeline finished WITH FAILURES. See ${LOG_DIR}/ and ${FAILED_FILE}"
    exit 1
fi
echo "[+] MIA pipeline complete! CSVs in ${OUT_DIR}/ and plots/tables in ${PLOT_DIR}/"
