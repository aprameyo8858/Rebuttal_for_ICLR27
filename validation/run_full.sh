#!/bin/bash
set -e

EVAL_SIZE=${1:-50000}
CW_BATCH_SIZE=${2:-512}
NORM_LAYER=${3:-lnchw}          

GPUS=(0)
NUM_GPUS=${#GPUS[@]}
DATASETS=("cifar10" "cifar100")
MODELS=("resnet50" "resnet34" "resnet18")
SEEDS=(42 43 44)

OUT_DIR="./results"
PLOT_DIR="./plots"
DATA_DIR="./data"
LOG_DIR="${OUT_DIR}/logs_full"

mkdir -p "$OUT_DIR" "$PLOT_DIR" "$DATA_DIR" "$LOG_DIR"
export PYTHONUNBUFFERED=1

if [ "$NORM_LAYER" != "lnchw" ]; then
    echo "[!] WARNING: NORM_LAYER=${NORM_LAYER}. Only 'lnchw' has been validated ('ln' diverged)."
fi

echo "=================================================================="
echo " Starting ${NUM_GPUS}-GPU Parallel Full Pipeline"
echo " GPUs: ${GPUS[*]} | C&W Eval Size: ${EVAL_SIZE} | C&W Batch Size: ${CW_BATCH_SIZE} | Norm: ${NORM_LAYER}"
echo "=================================================================="

python -c "
from torchvision import datasets
datasets.CIFAR10(root='${DATA_DIR}', train=True, download=True)
datasets.CIFAR10(root='${DATA_DIR}', train=False, download=True)
datasets.CIFAR100(root='${DATA_DIR}', train=True, download=True)
datasets.CIFAR100(root='${DATA_DIR}', train=False, download=True)
"

JOBS=()
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            JOBS+=("${DATASET}:${MODEL}:${SEED}")
        done
    done
done
TOTAL_JOBS=${#JOBS[@]}

# NEW: remove stale per-seed / merged CSVs so a failed job can never be merged with old (e.g. diverged) results.
# They are regenerated from the checkpoints (valid checkpoints are reused, so nothing is retrained).
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        rm -f "${OUT_DIR}/full_${DATASET}_${MODEL}_3seeds.csv" "${OUT_DIR}/full_all50k_dynamics_${DATASET}_${MODEL}_3seeds.csv"
        for SEED in "${SEEDS[@]}"; do
            rm -f "${OUT_DIR}/full_${DATASET}_${MODEL}_seed${SEED}.csv" "${OUT_DIR}/full_all50k_dynamics_${DATASET}_${MODEL}_seed${SEED}.csv"
        done
    done
done

echo "[*] Dispatching ${TOTAL_JOBS} total seed runs across ${NUM_GPUS} GPUs..."

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=0
done

for JOB_IDX in "${!JOBS[@]}"; do
    IFS=":" read -r DATASET MODEL SEED <<< "${JOBS[$JOB_IDX]}"
    ASSIGNED_SLOT=-1
    while [ $ASSIGNED_SLOT -eq -1 ]; do
        for ((g=0; g<NUM_GPUS; g++)); do
            PID=${GPU_PIDS[$g]}
            if [ "$PID" -eq 0 ] || ! kill -0 "$PID" 2>/dev/null; then
                ASSIGNED_SLOT=$g
                break
            fi
        done
        if [ $ASSIGNED_SLOT -eq -1 ]; then
            sleep 2
        fi
    done

    GPU_ID=${GPUS[$ASSIGNED_SLOT]}
    LOG_FILE="${LOG_DIR}/full_${DATASET}_${MODEL}_seed${SEED}_gpu${GPU_ID}.log"
    echo "[$(date '+%H:%M:%S')] [Job $((JOB_IDX+1))/${TOTAL_JOBS}] Launching ${DATASET} | ${MODEL} | Seed ${SEED} on GPU ${GPU_ID} -> ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=$GPU_ID python train_eval_full.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --norm-layer "$NORM_LAYER" \
        --seeds "$SEED" \
        --target-seeds-for-merge "${SEEDS[@]}" \
        --epochs 200 \
        --batch-size 128 \
        --cw-batch-size "$CW_BATCH_SIZE" \
        --eval-size "$EVAL_SIZE" \
        --lr 0.1 \
        --data-dir "$DATA_DIR" \
        --out-dir "$OUT_DIR" > "$LOG_FILE" 2>&1 &

    GPU_PIDS[$ASSIGNED_SLOT]=$!
done

echo "[*] Waiting for all active GPU jobs to finish..."
wait

# NEW: background exit codes are invisible to 'set -e', so check which seed runs actually produced results.
N_FAILED=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            if [ ! -f "${OUT_DIR}/full_${DATASET}_${MODEL}_seed${SEED}.csv" ]; then
                echo "[!] FAILED: ${DATASET} | ${MODEL} | seed ${SEED}  (see ${LOG_DIR}/full_${DATASET}_${MODEL}_seed${SEED}_gpu*.log)"
                N_FAILED=$((N_FAILED + 1))
            fi
        done
    done
done
if [ "$N_FAILED" -eq 0 ]; then
    echo "[+] All ${TOTAL_JOBS} training & C&W evaluation jobs finished!"
else
    echo "[!] ${N_FAILED}/${TOTAL_JOBS} jobs failed. Re-running this script resumes them (valid checkpoints are reused)."
fi

echo "=================================================================="
echo " Merging 3-Seed CSVs, Generating Verification Plots & Printing Scores"
echo "=================================================================="

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "resnet18" "resnet34" "resnet50"; do
        python train_eval_full.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --target-seeds-for-merge "${SEEDS[@]}" \
            --out-dir "$OUT_DIR" \
            --merge-only || echo "[!] Merge failed for ${DATASET}/${MODEL}"

        FULL_CSV="${OUT_DIR}/full_${DATASET}_${MODEL}_3seeds.csv"
        MIA_CSV="${OUT_DIR}/mia_${DATASET}_${MODEL}_3seeds.csv"

        if [ ! -f "$FULL_CSV" ]; then
            echo "[!] Skipping plots for ${DATASET}/${MODEL}: merged CSV missing (a seed failed)."
            continue
        fi

        if [ -f "$MIA_CSV" ]; then
            python plot_verification.py \
                --full-csv "$FULL_CSV" \
                --mia-csv "$MIA_CSV" \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --out-dir "$PLOT_DIR" || echo "[!] Plotting failed for ${DATASET}/${MODEL}"
        else
            python plot_verification.py \
                --full-csv "$FULL_CSV" \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --out-dir "$PLOT_DIR" || echo "[!] Plotting failed for ${DATASET}/${MODEL}"
        fi
    done
done

echo "[+] Full pipeline complete! CSVs in ${OUT_DIR}/ and plots/tables in ${PLOT_DIR}/"
