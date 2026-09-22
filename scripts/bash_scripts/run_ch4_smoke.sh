#!/usr/bin/env bash
# Chapter 4 Phase 2 smoke (docs/ch4_conditional_adaptation_plan.md Section 5).
#
# Every run writes its own log under logs/ch4/smoke/ via tee, so a crash leaves
# the traceback on disk instead of losing it with the pipeline.  Runs that
# already produced a summary.json are skipped, so the script is resumable.
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-$HOME/.conda/envs/irr/bin/python}
START=0615
DAYS=3
# Poznan carries a single verification run rather than the full set (author
# decision 2026-09-22): the smoke proves the site loads and scores, while the
# method comparison is exercised at Folsom and Karpicko.
SITES=(folsom karpicko)
VERIFY_SITE=poznan
OUT=results/ch4/smoke
LOG=logs/ch4/smoke
mkdir -p "$LOG"
FAIL=$LOG/failures.log
: > "$FAIL"

run() {            # run <name> <out-subdir> <cmd...>
  local name=$1 dir=$2; shift 2
  if [[ -f "$OUT/$dir/summary.json" ]]; then
    echo "skip   $name (already complete)"; return 0
  fi
  echo "run    $name"
  "$@" 2>&1 | tee "$LOG/$name.log" >/dev/null
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "FAILED $name (exit $rc) — see $LOG/$name.log" | tee -a "$FAIL"
  fi
  return 0
}

COND=scripts/related_works/evaluate_lagged_ogd.py
SLID=scripts/related_works/evaluate_sliding_window.py
SMRLS=scripts/related_works/evaluate_sm_rls.py
EVAL=scripts/evaluate.py

for site in "${SITES[@]}"; do
  d=data/${site}_prepared
  common=(--dataset-path "$d" --start-day $START --num-days $DAYS --seed 0)

  # references and comparators
  run "${site}_static"   "$site/baseline/seed0"       $PY $EVAL "${common[@]}" --method none  --out-dir "$OUT/$site/baseline/seed0"
  run "${site}_note"     "$site/note_reg_all/seed0"   $PY $EVAL "${common[@]}" --method note  --out-dir "$OUT/$site/note_reg_all/seed0"
  run "${site}_memo"     "$site/memo_reg_all/seed0"   $PY $EVAL "${common[@]}" --method memo  --out-dir "$OUT/$site/memo_reg_all/seed0"
  run "${site}_fewshot1" "$site/fewshot_d1/seed0"     $PY $EVAL "${common[@]}" --method fewshot --fewshot-days 1 --out-dir "$OUT/$site/fewshot_d1/seed0"
  run "${site}_sliding"  "$site/sliding_window/seed0" $PY $SLID "${common[@]}" --out-dir "$OUT/$site/sliding_window/seed0"
  run "${site}_smrls"    "$site/sm_rls/seed0"         $PY $SMRLS "${common[@]}" --out-dir "$OUT/$site/sm_rls/seed0"

  # few-shot 7d needs more days than the smoke window has (see the manifest)
  run "${site}_fewshot7" "$site/fewshot_d7/seed0" $PY $EVAL --dataset-path "$d" \
      --start-day $START --num-days 10 --seed 0 --method fewshot --fewshot-days 7 \
      --out-dir "$OUT/$site/fewshot_d7/seed0"

  # the proposed policy; retention needs a source stream other than the site itself
  if [[ $site == folsom ]]; then
    run "${site}_cond" "$site/conditional/seed0" $PY $COND "${common[@]}" --no-retention \
        --out-dir "$OUT/$site/conditional/seed0"
  else
    run "${site}_cond" "$site/conditional/seed0" $PY $COND "${common[@]}" \
        --retention-source data/folsom_prepared --retention-early-days 2 --retention-source-days 2 \
        --out-dir "$OUT/$site/conditional/seed0"
  fi
done

# Poznan: one configuration, the section's own policy, which exercises the two
# update actions, the events schema, the clamp sentinel and the retention pass.
run "${VERIFY_SITE}_cond" "$VERIFY_SITE/conditional/seed0" $PY $COND \
    --dataset-path "data/${VERIFY_SITE}_prepared" --start-day $START --num-days $DAYS --seed 0 \
    --retention-source data/folsom_prepared --retention-early-days 2 --retention-source-days 2 \
    --out-dir "$OUT/$VERIFY_SITE/conditional/seed0"

# S3 determinism: repeat seed 0 and run seed 1, one site is enough
run "karpicko_cond_seed0_repeat" "karpicko/conditional/seed0_repeat" $PY $COND \
    --dataset-path data/karpicko_prepared --start-day $START --num-days $DAYS --seed 0 --no-retention \
    --out-dir "$OUT/karpicko/conditional/seed0_repeat"
run "karpicko_memo_seed1" "karpicko/memo_reg_all/seed1" $PY $EVAL \
    --dataset-path data/karpicko_prepared --start-day $START --num-days $DAYS --seed 1 --method memo \
    --out-dir "$OUT/karpicko/memo_reg_all/seed1"

# S9 regression: published population (no daylight filter) and published checkpoint.
# Metrics cannot match — the published run adapted from 0101 and its high-error
# action was a head update (plan R13) — so this pins the population only.
run "karpicko_regression" "karpicko/regression_legacy/seed0" $PY $COND \
    --dataset-path data/karpicko_prepared --start-day $START --num-days $DAYS --seed 0 --no-retention \
    --min-ghi-clear 0 --checkpoint checkpoints/epoch_010.ckpt \
    --out-dir "$OUT/karpicko/regression_legacy/seed0"

echo
if [[ -s "$FAIL" ]]; then echo "FAILURES:"; cat "$FAIL"; else echo "all runs completed"; fi
