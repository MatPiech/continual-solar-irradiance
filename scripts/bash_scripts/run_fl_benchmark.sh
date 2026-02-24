#!/bin/bash
# FL Benchmark — run all five algorithms across a configurable list of datasets.
#
# Usage:
#   bash scripts/bash_scripts/run_fl_benchmark.sh [SYNC_INTERVAL] [NUM_DAYS] [DATASET ...]
#
# Arguments (all optional):
#   SYNC_INTERVAL : days between FL rounds (default: 7)
#   NUM_DAYS      : days to simulate, -1 = all (default: -1)
#   DATASET ...   : two or more dataset names (default: folsom karpicko poznan)
#                   any dataset with a matching data/<name>_prepared directory is valid
#
# Examples:
#   bash scripts/bash_scripts/run_fl_benchmark.sh                          # all defaults
#   bash scripts/bash_scripts/run_fl_benchmark.sh 14 365                   # bi-weekly, full year
#   bash scripts/bash_scripts/run_fl_benchmark.sh 7 30 folsom karpicko     # smoke-test two datasets
#   bash scripts/bash_scripts/run_fl_benchmark.sh 7 -1 folsom karpicko poznan

set -e
pyenv local irr

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
SYNC_INTERVAL="${1:-7}"          # FL aggregation every N days
NUM_DAYS="${2:--1}"              # -1 = all available days

# All datasets to include as clients in a single N-client run.
# Pass extra positional args to override.
if [[ $# -ge 3 ]]; then
    shift 2
    DATASETS=("$@")
else
    DATASETS=(folsom karpicko poznan)
fi

# FL algorithms to run for every dataset pair (comment out any you want to skip).
ALGORITHMS=(
    isolated   # lower bound — no federation
    fedavg     # standard weight averaging
    fedprox    # FedAvg + proximal term
    fedbn      # FedAvg, BatchNorm layers kept local
    fedper     # backbone aggregated, head kept local
)

CHECKPOINT="checkpoints/epoch_011.ckpt"
RESULTS_ROOT="results/fl_benchmark/sync${SYNC_INTERVAL}"

# Shared local-training hyperparameters (naive fine-tuning — intentionally
# minimal so that FL aggregation does the heavy lifting).
ADAPT_LR=1e-5
ADAPT_WD=1e-3
ADAPT_GRAD_NORM=1.0
ADAPT_BATCH=32
ADAPT_EPOCHS=1
FEDPROX_MU=0.01
SEED=42
DEVICE="cuda"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_algorithm() {
    local ALGO="$1"
    local OUT_DIR="${RESULTS_ROOT}/${ALGO}"

    # Build --dataset flags for every dataset
    local DATASET_FLAGS=()
    for DS in "${DATASETS[@]}"; do
        DATASET_FLAGS+=(--dataset "data/${DS}_prepared")
    done

    echo ""
    echo "========================================================================"
    echo "  Algorithm : ${ALGO^^}"
    echo "  Datasets  : ${DATASETS[*]}"
    echo "  Sync      : every ${SYNC_INTERVAL} days"
    echo "  Days      : ${NUM_DAYS} (-1 = all)"
    echo "  Output    : ${OUT_DIR}"
    echo "========================================================================"

    python scripts/fl_benchmark.py \
        --checkpoint          "${CHECKPOINT}" \
        "${DATASET_FLAGS[@]}" \
        --algorithm           "${ALGO}" \
        --sync-interval       "${SYNC_INTERVAL}" \
        --num-days            "${NUM_DAYS}" \
        --adapt-lr            "${ADAPT_LR}" \
        --adapt-weight-decay  "${ADAPT_WD}" \
        --adapt-max-grad-norm "${ADAPT_GRAD_NORM}" \
        --fedprox-mu          "${FEDPROX_MU}" \
        --adapt-batch-size    "${ADAPT_BATCH}" \
        --adapt-epochs        "${ADAPT_EPOCHS}" \
        --device              "${DEVICE}" \
        --seed                "${SEED}" \
        --save-results        "${OUT_DIR}"

    echo "  Done. Results saved to ${OUT_DIR}/"
}

# ---------------------------------------------------------------------------
# Validate global inputs
# ---------------------------------------------------------------------------
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

for DS in "${DATASETS[@]}"; do
    if [[ ! -d "data/${DS}_prepared" ]]; then
        echo "ERROR: Dataset not found: data/${DS}_prepared"
        exit 1
    fi
done

echo "========================================================================"
echo "  FL Benchmark"
echo "  Datasets   : ${DATASETS[*]}"
echo "  Algorithms : ${ALGORITHMS[*]}"
echo "  Sync       : every ${SYNC_INTERVAL} days"
echo "  Days       : ${NUM_DAYS} (-1 = all)"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Results    : ${RESULTS_ROOT}/"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Main loop — one N-client run per algorithm
# ---------------------------------------------------------------------------
for ALGO in "${ALGORITHMS[@]}"; do
    run_algorithm "${ALGO}"
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "  All runs completed.  Final MAE summary:"
echo "  $(printf '%-12s  %s' 'algorithm' 'MAE (all clients)')"
echo "  $(printf '%-12s  %s' '---------' '-----------------')"
find "${RESULTS_ROOT}" -name "summary.json" | sort | while read -r f; do
    ALGO=$(basename "$(dirname "$f")")
    MAES=$(python -c "
import json; d=json.load(open('$f'))
parts = [f\"{name}={m.get('mae','N/A'):.4f}\" for name, m in d.get('overall_metrics', {}).items()]
print('  '.join(parts))
" 2>/dev/null || echo "N/A")
    printf "  %-12s  %s\n" "${ALGO}" "${MAES}"
done
echo "========================================================================"
