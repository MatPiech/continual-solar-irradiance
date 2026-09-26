#!/usr/bin/env bash
# Hardware evaluation, priorities 1-3, on one device (run inside tmux ON the device).
#
#     bash scripts/bash_scripts/run_edge_device.sh <pi5|orin>
#
#   1  Chapter 4 adaptation: per-action latency, energy and peak memory for every method   (edge_adaptation_benchmark.py)
#   2  Chapter 5 collaboration: isolated / fedavg / prototypes per phase on 7 real days     (edge_collaboration_benchmark.py)
#   3  Chapter 4 admissibility: exclusion-flag cost per frame and net cost per day          (edge_admissibility_cost.py)
#
# Before starting:
#   - repository at the same commit as the host, conda env with torch (set PY=... otherwise)
#   - checkpoints/resnet50_epoch_015.ckpt and results/adaptation/gate/local/local_gate.npz copied over
#   - Folsom subset for step 2: data/edge/folsom_edge_subset.tar.gz (461 MB; periods.pickle, folsom_clear_sky.csv
#     and the 5622 images of the 8 selected days), unpacked on the device with
#       tar -C data -xzf folsom_edge_subset.tar.gz      # -> data/folsom_prepared/
#   - Orin: max power mode set and fan on; Pi 5: active cooler fitted, nothing else running
#   - inline meter logging (optional cross-check of the PMIC / tegrastats energy)
#
# Each step is teed to its own log; a failed step is reported and the next one still runs.
set -uo pipefail
cd "$(dirname "$0")/../.."

DEV=${1:?usage: run_edge_device.sh <pi5|orin>}
case $DEV in
  pi5)  DEVICE=cpu;  THREADS=${THREADS:-4} ;;
  orin) DEVICE=cuda; THREADS=${THREADS:-6} ;;
  *) echo "device must be pi5 or orin (got '$DEV')" >&2; exit 2 ;;
esac
PY=${PY:-$HOME/.conda/envs/irr/bin/python}
SOAK=${SOAK:-600}           # thermal soak before timing, seconds
SAMPLES=${SAMPLES:-200}     # timed repetitions per per-action measurement
DAYS=${DAYS:-7}             # timed days per Chapter 5 arm
COOL_TO=${COOL_TO:-50}      # chip temperature (C) waited for before every method, arm, timed block and step
OUT=results/edge/$DEV
LOG=logs/edge/$DEV
mkdir -p "$OUT" "$LOG"
export OMP_NUM_THREADS=$THREADS

step() {  # step <name> <command...>
  local name=$1; shift
  [[ $name != environment && $name != soak ]] && "$PY" scripts/edge_power.py cooldown "$COOL_TO"
  echo "=== $name  $(date -Is)"
  "$@" 2>&1 | tee "$LOG/$name.log"
  local rc=${PIPESTATUS[0]}
  [[ $rc -ne 0 ]] && echo "FAILED $name (exit $rc), see $LOG/$name.log" | tee -a "$LOG/failures.log"
  return 0
}

step environment bash -c "cat /proc/device-tree/model 2>/dev/null; echo; uname -a; free -m; \
  (command -v nvpmodel >/dev/null && nvpmodel -q) || true; (command -v vcgencmd >/dev/null && vcgencmd get_throttled && vcgencmd measure_temp) || true; \
  $PY -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'; git rev-parse HEAD"

step soak "$PY" - <<EOF
import time, torch
from cirr.evaluation.runner import load_eval_model
torch.set_num_threads($THREADS)
m = load_eval_model("checkpoints/resnet50_epoch_015.ckpt", "$DEVICE")
x, h = torch.rand(1, 3, 128, 128, device="$DEVICE"), torch.rand(1, 4, device="$DEVICE")
end, n = time.time() + $SOAK, 0
with torch.no_grad():
    while time.time() < end:
        m(x, h); n += 1
print(f"soak: {n} forward passes in $SOAK s")
EOF

step adaptation "$PY" scripts/edge_adaptation_benchmark.py --checkpoint checkpoints/resnet50_epoch_015.ckpt \
  --device "$DEVICE" --threads "$THREADS" --samples "$SAMPLES" --cool-to "$COOL_TO" --out "$OUT/adaptation.json"

step collaboration "$PY" scripts/edge_collaboration_benchmark.py --dataset data/folsom_prepared --days "$DAYS" \
  --device "$DEVICE" --threads "$THREADS" --cool-to "$COOL_TO" --out "$OUT/collaboration.json"

step admissibility "$PY" scripts/edge_admissibility_cost.py --device "$DEVICE" --threads "$THREADS" \
  --samples "$SAMPLES" --cool-to "$COOL_TO" --overhead-json "$OUT/adaptation.json" --out "$OUT/admissibility.json"

(command -v vcgencmd >/dev/null && echo "throttling flags at end: $(vcgencmd get_throttled)") || true
echo "=== done $(date -Is); results in $OUT/, logs in $LOG/"
[[ -s "$LOG/failures.log" ]] && cat "$LOG/failures.log" || echo "no failures"
