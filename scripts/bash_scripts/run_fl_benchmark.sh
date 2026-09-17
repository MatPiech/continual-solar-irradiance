#!/bin/bash
# FL Benchmark — run selected algorithms across a configurable list of datasets.
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
    static     # inference-only baseline — no training, no aggregation
    isolated   # do-no-harm reference — no federation
    fedavg     # standard weight averaging
    fedbn      # FedAvg, BatchNorm layers kept local
    fedper     # backbone aggregated, head kept local
    fedprox    # FedAvg + proximal term in local training
    fedema     # FedAvg + EMA interpolation at distribution
    fedproto   # prototype exchange, no weight sharing (the proposed method)
)

# Chapter 5 runs on the explicitly named ResNet50 checkpoint so that the model
# matches the architecture described in the thesis.  Switch to
# checkpoints/efficientvit_b2_epoch_012.ckpt only together with the text.
CHECKPOINT="checkpoints/resnet50_epoch_015.ckpt"

# Shared local-training hyperparameters (naive fine-tuning — intentionally
# minimal so that FL aggregation does the heavy lifting).
WARMUP_DAYS=30
WARMUP_CACHE_DIR="warmup_cache"
ADAPT_LR=1e-5
ADAPT_WEIGHT_DECAY=1e-4
ADAPT_GRAD_NORM=1.0
ADAPT_BATCH=32
ADAPT_EPOCHS=1
# The Folsom reservoir buffer is OFF: it placed source-site raw samples on every
# client, which contradicts the premise that measurements never leave a site.
# Only each client's own recent history is replayed.
REPLAY_BUFFER_SIZE=0       # reservoir buffer capacity (0 = disabled)
REPLAY_RATIO=0.5           # reservoir samples as fraction of current-day count
RECENT_BUFFER_SIZE=1000    # FIFO recent buffer capacity (0 = disabled)
RECENT_REPLAY_RATIO=0.5    # recent-buffer samples as fraction of current-day count
# Baseline strength parameters.  A negative result about parameter-space
# federation is only as strong as the effort spent making those baselines work,
# so each is set to its registered / published-default value rather than to the
# most restrictive end of its range.  Tier 3 sweeps them; the main table reports
# each baseline at its best setting.
FEDPROX_MU=0.01            # proximal strength for FedProx  (sweep: 0.001 0.01 0.1 1.0)
DITTO_LAMBDA=0.1           # personalised proximal strength (sweep: 0.01 0.1 1.0)
FEDEMA_ALPHA=0.5           # EMA coefficient, higher keeps more local (sweep: 0.1 0.5 0.9)
FEDPROTO_LAMBDA=0.1        # prototype regularisation strength for FedProto
SEED=42
REGIME_LABEL="clear_sky"   # clear-sky index rather than the fixed-1000 proxy
DEVICE="cuda"
PARALLEL_CLIENTS="${#DATASETS[@]}"  # number of datasets

RESULTS_ROOT="results/fl_benchmark/days${NUM_DAYS}_warmup${WARMUP_DAYS}_sync${SYNC_INTERVAL}"

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
        --warmup-days         "${WARMUP_DAYS}" \
        ${WARMUP_CACHE_DIR:+--warmup-cache-dir "${WARMUP_CACHE_DIR}"} \
        --replay-buffer-size  "${REPLAY_BUFFER_SIZE}" \
        --replay-ratio        "${REPLAY_RATIO}" \
        --recent-buffer-size  "${RECENT_BUFFER_SIZE}" \
        --recent-replay-ratio "${RECENT_REPLAY_RATIO}" \
        --adapt-lr            "${ADAPT_LR}" \
        --adapt-weight-decay  "${ADAPT_WEIGHT_DECAY}" \
        --adapt-max-grad-norm "${ADAPT_GRAD_NORM}" \
        --adapt-batch-size    "${ADAPT_BATCH}" \
        --adapt-epochs        "${ADAPT_EPOCHS}" \
        --fedprox-mu          "${FEDPROX_MU}" \
        --ditto-lambda        "${DITTO_LAMBDA}" \
        --fedema-alpha        "${FEDEMA_ALPHA}" \
        --fedproto-lambda     "${FEDPROTO_LAMBDA}" \
        --regime-label        "${REGIME_LABEL}" \
        --daylight-filter \
        --device              "${DEVICE}" \
        --seed                "${SEED}" \
        --parallel-clients    "${PARALLEL_CLIENTS:-${#DATASETS[@]}}" \
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
    # Skip results from algorithms not in the current ALGORITHMS array
    [[ " ${ALGORITHMS[*]} " == *" ${ALGO} "* ]] || continue
    MAES=$(python -c "
import json; d=json.load(open('$f'))
parts = [f\"{name}={m.get('mae','N/A'):.4f}\" for name, m in d.get('overall_metrics', {}).items()]
print('  '.join(parts))
" 2>/dev/null || echo "N/A")
    printf "  %-12s  %s\n" "${ALGO}" "${MAES}"
done
echo "========================================================================"
