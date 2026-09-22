# Chapter 4 Phase 2 — Smoke benchmark report

Branch: `feature/adaptation` · Repository: `continual-solar-irradiance` · Date: 2026-09-22
Scope: `docs/ch4_conditional_adaptation_plan.md` §5. Twenty runs on a 3-day window.
Hardware: NVIDIA GeForce RTX 4090 (24 GiB), 32 CPU cores, environment `~/.conda/envs/irr`.

**Verdict: cleared for Tier 1.** Every configuration runs end to end, the schema is complete at
every run, and the two update actions of Section 4.3 both fire at every site. All eleven checks
S1–S11 pass. The smoke found one defect that would have silently corrupted the campaign's
forecast-skill column; it is fixed, the affected artifacts are recomputed, and a check now guards
it.

Two results that belong to the science rather than the mechanics appear below and should be read
before Tier 1 is scoped: the full-model action fires on roughly a quarter of updates (§6.1), and
three days of adaptation already cost 32–36 % on the source stream (§6.2).

## 1. What was run

Twenty runs under `results/ch4/smoke/`, launched by `scripts/bash_scripts/run_ch4_smoke.sh`, one
tee'd log each under `logs/ch4/smoke/`. `failures.log` is empty.

| Site | Configurations |
|---|---|
| Folsom | static, MEMO, NOTE, SM-RLS, sliding window, few-shot 1d, few-shot 7d, conditional |
| Karpicko | the same eight, plus the S3 determinism pair and the S9 population regression |
| Poznań | conditional only |

Poznań was cut to a single verification run by author decision during the sweep. The conditional
policy is the configuration that exercises the most machinery — both update actions, the events
schema, the clamp sentinel and the retention pass — so the site is proved to load and score
without repeating the method comparison a third time.

## 2. Window (plan §5.1)

**15–17 June**, 7,873 lit samples, recorded in `configs/ch4_smoke_window.yaml`.

| Site | samples | worst day coverage | mean |
|---|---|---|---|
| Folsom | 2,473 | 0.919 | 0.950 |
| Karpicko | 2,523 | 0.829 | 0.868 |
| Poznań | 2,877 | 0.965 | 0.987 |

The plan inherited a rule from Chapter 5 — first 7 consecutive days, March to May, ≥80 % coverage
everywhere — that has no solution on these sites. The best March-to-May window, 9–15 May, reaches
0.61 at Karpicko, which additionally holds no data for 1–20 April. Fourteen strictly consecutive
windows clear 80 %; 15–21 June is the largest, and the smoke uses its first three days.

**Few-shot 7d cannot run on a 3-day window.** `evaluate_fewshot` fits on the first `fewshot_days`
days and predicts on the remainder, so three days leave it nothing to predict. It runs on 15–24
June for the executability check only, and is excluded from every check that compares populations.
Its metrics are not comparable with the other rows.

## 3. Acceptance checks S1–S11

Produced by `scripts/check_ch4_smoke.py`; full output in `logs/ch4/check_smoke.log`.

| ID | Check | Result | Evidence |
|---|---|---|---|
| S1 | Completion | **PASS** | 20 of 20 runs wrote `summary.json`; `failures.log` empty |
| S2 | Sanity | **PASS** | worst daily MAE 220.2 W/m² (Karpicko few-shot 7d), limit 300; no non-finite values |
| S3 | Determinism | **PASS** | conditional seed 0 twice byte-identical; MEMO seed 0 vs 1 differ (0.08007157 vs 0.08010066) |
| S4 | Daylight filter | **PASS** | no scored sample below the filter; every run matches the manifest population exactly |
| S5 | Action accounting across methods | **PASS** | both actions fire at all three sites; MEMO and NOTE `adapt_events` equal their `events.csv` totals |
| S6 | Action accounting | **PASS** | `events.csv` reconciles with per-sample rows everywhere; `scope_changed_if_raw` zero in all 20 runs |
| S7 | Static reference | **PASS** | static pooled MAE finite and sane at both comparison sites |
| S8 | Smart Persistence | **PASS** | `sp_rmse` identical across all methods per site, after the §5 fix |
| S9 | Population regression | **PASS** | per-day counts reproduce the published run exactly: 932 / 919 / 941 |
| S10 | Logs | **PASS** | all five files and every required column present in 20 runs |
| S11 | Runtime | **PASS** | 25.7 min for 20 runs; worst configuration MEMO at 3.3 h per campaign site (§7) |

## 4. S9 in detail — what the regression check can and cannot show

Run with `--min-ghi-clear 0` and `--checkpoint checkpoints/epoch_010.ckpt`, reproducing the
published configuration on the same three days.

| day | published | re-run |
|---|---|---|
| 0615 | 932 | 932 |
| 0616 | 919 | 919 |
| 0617 | 941 | 941 |

The population reproduces exactly, which is what this check is for. **Metrics are deliberately not
compared**, for two independent reasons. The published run reached 15 June having adapted
continuously since 1 January, while this one starts from the frozen checkpoint. And its high-error
action was a head update, not a full-model update (plan R13), so it is a different policy. Those
269 extra samples per three days — 9.6 % of the June population at Karpicko — are the night frames
the Chapter 5 filter removes; in June the Polish nights are at their shortest, and the full-year
share is 20.5 %.

## 5. Defect found and fixed

**Smart Persistence was scored in mixed units.** `smart_persistence_daily` called `predict_periods`
without `period_scale`. Dataset periods carry irradiance in W/m² while the clear-sky lookup is in
kW/m², so the clear-sky index came out a factor of 1000 too large, saturated at its clamp of 2.0,
and the resulting "RMSE" mixed the two units. Every `sp_rmse` value and every `forecast_skill_pct`
in the first sweep was wrong — Folsom read 568 W/m² and a forecast skill of 99.98 %.

Chapter 5's `scripts/compute_sp_metrics.py` passes the scale correctly, which is how the
discrepancy was visible at all. Fixed in `cirr/evaluation/campaign.py` with the reason recorded at
the call site. Smart Persistence depends only on the scored population, so the affected artifacts
were recomputed rather than re-run, which is exactly equivalent. After the fix:

| site | Smart Persistence RMSE |
|---|---|
| Folsom | 143.0 W/m² |
| Karpicko | 102.5 W/m² |
| Poznań | 134.9 W/m² |

S8 now also rejects a pooled Smart Persistence RMSE outside 10–500 W/m² and a forecast skill
outside (−200, 100) %, so this class of unit error cannot pass silently again.

A second, smaller defect was in the checker itself rather than the pipeline: the `head` action
column shadows `DataFrame.head`, so attribute access raised instead of summing. Every action
column is now bracket-accessed.

## 6. Two findings for Tier 1 scoping

### 6.1 The full-model action fires, and often

| site | no update | head | full model |
|---|---|---|---|
| Folsom | 389 | 1,376 | 663 |
| Karpicko | 346 | 1,557 | 575 |
| Poznań | 486 | 1,747 | 599 |

Roughly 22–24 % of considered samples take the full-model action. Before Phase 1 that action
trained only the head (plan R13), so the published campaign never paid its cost. This is the first
evidence of what the policy as the chapter describes it actually costs, and it means the device
benchmark in `tab:ch4-solar-edge-benchmark` measures a cheaper policy than the one now implemented.

### 6.2 Three days of adaptation already cost 32–36 % on the source stream

The retention pass re-scores the adapted model on a held-out Folsom slice against the frozen
reference:

| deployment site | frozen Folsom MAE | adapted Folsom MAE | change |
|---|---|---|---|
| Karpicko | 30.51 W/m² | 41.38 W/m² | **+35.6 %** |
| Poznań | 30.51 W/m² | 40.12 W/m² | **+31.5 %** |

The chapter currently carries retention as "not measured" in its limitations. On this evidence it
is not merely unmeasured but substantial, and it appears after three days rather than a season.
The slice is two days and a single run, so this is a signal to size Tier 1 around, not a result:
plan hypothesis H4.4 sets the bar at 10 %, and the smoke is already three times past it.

The frozen reference of 30.51 W/m² is worth noting on its own, since the published
`tab:ch4-solar-benchmark` reports Folsom Static at 30.561 on the full year.

## 7. Runtime (S11) and what Tier 1 will cost

Total 25.7 minutes for 20 runs, after the few-shot data-loading fix described below.
Extrapolating to one 300-day site at ~800 lit samples/day needs two different cost models, and
conflating them is how an earlier draft of this report reached a figure ten times too high for
few-shot.

**Online methods pay per sample**, so their window rate scales directly:

| configuration | ms/sample | per campaign site |
|---|---|---|
| MEMO | 43–49 | 2.9–3.3 h |
| SM-RLS | 15–28 | 1.0–1.9 h |
| conditional | 18–23 | 1.2–1.5 h |
| sliding window | 18–21 | 1.2–1.4 h |
| NOTE | 12–15 | 0.8–1.0 h |
| static | 10–14 | 0.7–0.9 h |

**Few-shot fits once, offline, then only infers.** Its fit is fixed with respect to campaign
length — `fit_days = days_to_eval[:fewshot_days]`, so a 300-day campaign fits on the same 7 days a
10-day window does — and everything after it runs at the frozen model's inference rate.

| configuration | model | per campaign site |
|---|---|---|
| few-shot 7d (Karpicko) | fixed fit 3 min + inference at 14.0 ms/sample | 1.0 h |
| few-shot 7d (Folsom) | fixed fit 2 min + inference at 10.3 ms/sample | 0.7 h |
| few-shot 1d (Karpicko) | fixed fit 1 min + inference at 14.0 ms/sample | 0.9 h |
| few-shot 1d (Folsom) | fixed fit under a minute + inference at 10.3 ms/sample | 0.7 h |

Few-shot is among the cheapest configurations at campaign scale, not the most expensive.

### The few-shot fit was data-loading bound, and is not any more

`_fewshot_fit` never used a DataLoader. It indexed the datasets inline inside the training loop
and stacked each batch on the main thread, so every image was read, decoded with OpenCV,
transformed and normalised synchronously while the GPU waited. `load_dataset(workers=0)` was
never the lever, because the DataModule's loaders play no part in the fit.

The fit now builds one `DataLoader` over a small `(dataset, index)` pair wrapper — the fit pools
the deployment site with a Folsom replay buffer, so its samples come from two datasets — with
worker processes, `pin_memory`, and `persistent_workers` so the pool is not respawned each epoch.
Workers call `cv2.setNumThreads(0)`, since OpenCV's own pool otherwise makes the workers contend
for the same cores. `--fewshot-workers` defaults to 8 and is recorded in every run's config.

| run | before | after | speedup |
|---|---|---|---|
| few-shot 7d, Karpicko | 27.2 min (2 min 17 s per epoch) | **4.4 min** (14 s per epoch) | 6.2× |
| few-shot 1d, Karpicko | 4.3 min (20 s per epoch) | **1.1 min** (3 s per epoch) | 3.9× |

Whole-smoke wall time fell from 66.2 to 25.7 minutes.

Determinism is preserved: the per-epoch shuffle moved from an inline `random.Random` to the
DataLoader's seeded generator, and two runs at the same seed produce byte-identical
`metrics_daily.csv` and `predictions.csv`. The shuffle *order* changed, so few-shot metrics move
slightly against the pre-fix sweep — Karpicko few-shot 1d went from 81.72 to 83.58 W/m². All four
few-shot runs in this report were re-run after the fix, so every artifact here matches the
shipped code.

**Tier 1 estimate.** Taking Karpicko as the slower site: one seed each of static, SM-RLS, sliding
window and conditional is 5.7 h; three seeds each of MEMO, NOTE and the two few-shot variants is
18.6 h. That is about 24 h per site, **roughly 73 GPU-hours for three sites**, consistent with the
plan's 1–2 h per run. MEMO alone accounts for about 30 of those hours, so whether it needs three
seeds is the single biggest lever on Tier 1's cost.

Peak process RSS sits between 1.9 and 3.3 GB across every configuration, the highest being the
conditional policy at Karpicko. This is the quantity `sec:ch4-accelerator-execution` lists as
unmeasured for the accelerator path; it is now measured for the host path at campaign scale.

## 8. What the smoke does not establish

- **Nothing about accuracy.** Three days at one season, single seed for the deterministic methods.
  The per-method numbers in the run folders exist to prove the pipeline scores them, not to rank
  methods. In particular the conditional policy's apparent lead at Karpicko and Poznań rests on
  three days and must not be quoted.
- **H4.2 is untested**, and remains so by decision: the trigger controls were not implemented, so
  no evidence bears on whether the error threshold is active rather than inert.
- **Few-shot 7d is untested at smoke scope**, having run on a different window.
- **The device path is untouched.** Everything here is server-side.

## 9. Reproducing this phase

```bash
bash scripts/bash_scripts/run_ch4_smoke.sh          # resumable; skips completed runs
python scripts/check_ch4_smoke.py                   # S1-S11 table plus detail
```

Runs land under `results/ch4/smoke/<site>/<method>/seed<k>/`, logs under `logs/ch4/smoke/`.
