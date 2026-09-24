#!/usr/bin/env bash
# Proposed methods on the IEP (physics-informed, 7-channel) checkpoint, split over four servers.
#
#     CKPT=checkpoints/iep/iep_epoch_XXX.ckpt bash scripts/bash_scripts/run_physics_eval.sh <1|2|3|4>
#
# Minimal 2x2 design against the existing RGB results (results/ch4/tier1, results/fl/tier1/main):
#   Ch4, per site: static (no adaptation) vs conditional        — 6 runs
#   Ch5, 3 sites : FedAvg vs FedProto (lambda 0.1), seed 1234    — 2 runs
# Every protocol flag copies the RGB tier-1 run; only --checkpoint differs.  The
# dataset switches to the 7-channel input on its own (checkpoint hparams).
# A run whose summary.json exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."

SERVER=${1:?usage: run_physics_eval.sh <1|2|3|4>}
CKPT=${CKPT:?set CKPT to the IEP checkpoint}
PY=${PY:-$HOME/.conda/envs/irr/bin/python}
OUT=results/physics
LOG=logs/physics/eval
mkdir -p "$LOG"
FAIL=$LOG/failures_server${SERVER}.log
: > "$FAIL"

ch4() {  # ch4 <site> <static|conditional>
  local site=$1 method=$2 dir="$OUT/ch4/$1/$2/seed0" name="ch4_${1}_${2}"
  [[ -f "$dir/summary.json" ]] && { echo "skip   $name"; return 0; }
  local common=(--checkpoint "$CKPT" --dataset-path "data/${site}_prepared" --start-day 0101 --num-days -1 --seed 0 --out-dir "$dir")
  local cmd
  case $method in
    static)      cmd=("$PY" scripts/evaluate.py "${common[@]}" --method none) ;;
    conditional) cmd=("$PY" scripts/related_works/evaluate_lagged_ogd.py "${common[@]}" --no-retention) ;;
  esac
  echo "run    $name ($(date +%H:%M))"
  "${cmd[@]}" 2>&1 | tee "$LOG/$name.log" >/dev/null
  local rc=${PIPESTATUS[0]}
  [[ $rc -ne 0 ]] && echo "FAILED $name (exit $rc) — see $LOG/$name.log" | tee -a "$FAIL"
  return 0
}

ch5() {  # ch5 <fedavg|fedproto>
  local m=$1 dir="$OUT/ch5/$1/seed1234" name="ch5_$1"
  [[ -f "$dir/summary.json" ]] && { echo "skip   $name"; return 0; }
  local extra=()
  [[ $m == fedproto ]] && extra=(--fedproto-lambda 0.1)
  echo "run    $name ($(date +%H:%M))"
  # Pure FL configuration of Chapter 5 (warm-up 0, sync 1, no replay), as in run_ch5_tier1_main.sh.
  "$PY" scripts/fl_benchmark.py \
      --checkpoint "$CKPT" \
      --dataset data/folsom_prepared --dataset data/karpicko_prepared --dataset data/poznan_prepared \
      --start-day 0101 --num-days -1 \
      --warmup-days 0 --sync-interval 1 --warmup-cache-dir warmup_cache \
      --adapt-lr 1e-5 --adapt-weight-decay 1e-4 --adapt-max-grad-norm 1.0 \
      --adapt-batch-size 32 --adapt-epochs 1 \
      --replay-buffer-size 0 --recent-buffer-size 0 --recent-replay-ratio 0.5 \
      --regime-label clear_sky --device cuda --parallel-clients 3 \
      "${extra[@]}" --algorithm "$m" --seed 1234 --save-results "$dir" \
      2>&1 | tee "$LOG/$name.log" >/dev/null
  local rc=${PIPESTATUS[0]}
  [[ $rc -ne 0 || ! -f "$dir/summary.json" ]] && echo "FAILED $name (exit $rc) — see $LOG/$name.log" | tee -a "$FAIL"
  return 0
}

# Balanced on the RGB tier-1 wall times (1.5-1.8 h per slice); the 7-channel input
# decodes two images per sample, so expect roughly 2-2.5x that.
case $SERVER in
  1) ch5 fedproto; ch4 poznan static ;;
  2) ch5 fedavg; ch4 karpicko static ;;
  3) ch4 folsom conditional; ch4 folsom static ;;
  4) ch4 karpicko conditional; ch4 poznan conditional ;;
  *) echo "server index must be 1-4 (got '$SERVER')" >&2; exit 2 ;;
esac

echo "server $SERVER done"
if [[ -s "$FAIL" ]]; then echo "FAILURES:"; cat "$FAIL"; else echo "no failures"; fi
