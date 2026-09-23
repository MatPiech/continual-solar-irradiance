# Chapter 4 §4.3 benchmark plan for Claude Code

**Scope.** Section `sec:ch4-conditional-adaptation`, "Update justification under lagged
supervision". The sibling sections (`sec:ch4-admissibility`, `sec:ch4-accelerator-execution`)
are out of scope except where they share the evaluation population.

**How to use.** Open Claude Code in VS Code at the repository root and send:

```text
Read docs/ch4_conditional_adaptation_plan.md and execute Phase 0. Stop at Checkpoint A.
```

---

# Instructions for Claude Code

## 1. Role and ground rules

You are re-establishing the empirical basis of Chapter 4's conditional-adaptation section of
a doctoral dissertation. The section currently reports numbers whose scoring population,
source checkpoint and update accounting the chapter itself flags as unverified. Your job is
to make each reported quantity reproducible from the repository, then run the controls that
decide whether the error trigger is doing any work.

1. **Use `/ponytail` for all code work.** Invoke `/ponytail full` before you write, modify or
   refactor code in Phases 1 to 4, and again after any context compaction. Reading code and
   running existing commands do not need it. The reports this plan requests are explicitly
   asked for, so write them in full.
2. **Stop at every checkpoint.** At Checkpoints A, B, C and D, summarize and wait for approval.
3. **Do not invent values.** If the code does not answer a question, mark it **OPEN**.
4. **Protect existing results.** Work on branch `feature/adaptation`. Never overwrite or delete an
   existing folder under `results/`. New output goes to `results/ch4/`. Commit at the end of
   each phase with a message naming the phase.
5. **Reuse Chapter 5's machinery.** The federated campaign already solved the population,
   pooling and statistics problems this section has. Import, do not reimplement:
   `filter_lit_days` (`scripts/fl_benchmark.py:128`), `cirr/evaluation/smart_persistence.py`,
   `cirr/physics/clear_sky.py`, and the estimator stack in `cirr/fl/analysis.py`.
6. **No new dependencies without asking.**
7. **Long runs write to a log file with `tee`**, one log per run, and are filtered afterwards.
   Read `${PIPESTATUS[0]}` for the job's exit status.

## 2. Scientific context

### 2.1 What the section currently claims

| ID | Claim in the chapter | Evidence status today |
|---|---|---|
| C4.1 | Conditional adaptation reduces error under deployment shift (`tab:ch4-solar-benchmark`) | Single run, unfiltered population, 8 of 27 table cells backed by an artifact in `results/` |
| C4.2 | The policy trails sliding-window adaptation by ~1.4 % mean MAE | Same table, no interval, no common population |
| C4.3 | The policy adapts 2.5× faster and sustains 1.8× throughput (`tab:ch4-solar-edge-benchmark`) | Separate 640-sample stream, action mix unreported |
| C4.4 | Site-averaged MAE falls 29 / 50 / 61 % at H = 5 / 15 / 30 (`tab:ch4-solar-metrics`) | Week-scale streams not present in this repository |
| C4.5 | Updates concentrate on high-error intervals | Asserted from `fig:ch4-irradiance-plots`, never counted |

### 2.2 Hypotheses the re-run must test

| ID | Hypothesis | Pre-registered success criterion |
|---|---|---|
| H4.1 | The conditional policy is not worse than the strongest published comparator on a common daylight-filtered population | The 90 % CI of the relative MAE difference to sliding-window adaptation lies inside ±5 % at every site |
| H4.2 | *Deferred by author decision (2026-09-22).* The trigger controls are not implemented, so no evidence bears on whether the error threshold is active rather than inert. The chapter cannot claim the trigger does work until this is run |
| H4.3 | The update-rate advantage is real once the action mix is counted | Measured per-site action proportions give a campaign update cost below that of every gradient comparator at equal or better MAE |
| H4.4 | *Dropped by author decision (2026-09-23).* Retention is not needed for the chapter; runs use `--no-retention` |

## 3. Phase 0. Discovery and plan validation (read-only)

Do not modify any file in this phase.

### 3.1 Already resolved by the audit of 2026-09-22

These are settled. Re-verify each one cheaply, record the evidence, and do **not** re-open them.

| ID | Question the chapter marks unresolved | Answer | Evidence |
|---|---|---|---|
| R1 | Per-site proportions of no-update / head-update / full-update | Logged all along. Karpicko: considered 221 631, none 40.27 %, head 44.26 %, whole 15.47 % | `scripts/related_works/evaluate_lagged_ogd.py:55-90`; `results/karpicko/karpicko_paper_alg1_ogd_results.txt:13-17`; per-sample `adaptation_scope` column in the predictions CSV |
| R2 | Whether the head update recomputes features or reuses a cached embedding | Recomputes. The causal queue retains the **image tensor**, and `ogd_step` runs the full forward pass from the image | `cirr/evaluation/online.py:145-154`, `cirr/training/ogd.py:95-112` |
| R3 | Buffer memory | Deterministic: `horizon_steps` × (3 × 128 × 128 float32 image + 4-element history + scalars) ≈ 2.9 MB at H = 15. Compute and state it | `cirr/data/loading.py:8` (`DEFAULT_IMAGE_SIZE = [128, 128]`), `cirr/evaluation/online.py:36-41` |
| R4 | Feasibility of the full-model action from retained state | Feasible, because the image is retained. The implemented action was **not** `S = R^P` before Phase 1 — see R13 | `cirr/training/ogd.py:40-60,98` |
| R5 | Why MAE and WAPE rankings conflict | Different denominators. MAE/RMSE use every scored sample; WAPE/SMAPE use only targets ≥ 5 W/m². At Karpicko that is 190 943 of 226 114 samples | `cirr/utils/metrics.py:77-101` |
| R6 | Aggregation order behind the table | Pooled over all samples, not day-weighted. Karpicko conditional `Overall MAE 0.05481910` kW/m² reproduces the table's 54.819 exactly | `cirr/evaluation/online.py:176`; `results/karpicko/karpicko_paper_alg1_ogd_results.txt:322-327` |
| R7 | Whether night samples are scored | Yes, and the share differs sharply by site: Folsom 0 %, Karpicko 20.5 % (46 256 / 226 114), Poznań 32.5 % (59 521 / 183 044). No daylight filter exists anywhere in the Chapter 4 evaluation path | `cirr/evaluation/runner.py:34-49`; confirmed independently by the `filter_lit_days` docstring, `scripts/fl_benchmark.py:134-148` |
| R8 | Size of the night effect on the headline numbers | Restricting the existing Karpicko conditional run to lit samples moves MAE from **54.819 to 65.997 W/m²** (+20.4 %) and RMSE from 113.632 to 127.158. The near-parity with Smart Persistence (54.875) in `tab:ch4-solar-benchmark` is an artifact of the unfiltered population | Recomputed from `results/karpicko/karpicko_paper_alg1_ogd_predictions.csv` joined to the clear-sky lookup |
| R9 | Whether the policy spends updates on night frames | Yes. 26.19 % of dark frames trigger a head update and 8.38 % of all updates fire on dark frames. The trigger compares a ~13 W/m² night prediction against a ~0 W/m² target and exceeds T1 = 10 W/m² | Same recomputation; see the first rows of the Karpicko predictions CSV |
| R10 | The Chapter 4 / Chapter 5 Folsom Static discrepancy | Different source checkpoints. Every Chapter 4 result uses `checkpoints/epoch_010.ckpt` (md5 `991e1451…`); Chapter 5 uses `checkpoints/resnet50_epoch_015.ckpt` (md5 `21d5b347…`). Different files, not a rename | `grep '^Checkpoint:' results/*/*_results.txt`; `md5sum` of both |
| R11 | The MEMO scalar objective | Documented: prediction-variance minimisation across augmented views, not entropy. The chapter's description is wrong, not missing | `cirr/tta/memo_reg.py:1-31` |
| R14 | Whether the raw/clamped trigger divergence mattered | **No, and it is now closed.** Over the full published Karpicko year (221,631 considered samples) not one action differed: the minimum prediction is 0.41 W/m^2, the clamp never bound, and the raw/clamped gap is float32 rounding at most 4.5e-8 kW/m^2. Author decision 2026-09-22: trigger on the clamped forecast, because irradiance is non-negative throughout the data, so a negative output is not a forecast the device would emit. A per-run sentinel records what an unclamped trigger would have chosen | `scripts/related_works/evaluate_lagged_ogd.py` `run_policy`; `events.csv` column `scope_changed_if_raw` |
| R13 | Whether the full-model action was ever exercised | **No.** `set_trainable` unfroze `encoder_children[-2:]`, which on a timm backbone built with `num_classes=0` are `SelectAdaptivePool2d` and `Identity` — zero parameters. The high-error action trained the head and nothing else, so the published campaign ran a two-action policy (no-update, head-update) while the chapter describes three. Fixed in Phase 1; pre-Phase-1 results are not reproducible from the current code | `cirr/training/ogd.py` `set_trainable`; `tests/test_ch4_phase1.py::test_f6_the_published_high_error_action_trained_only_the_head` |
| R12 | The NOTE memory adaptation | Documented: the class-balanced reservoir is replaced by a plain FIFO deque, with the same prediction-variance objective and two BN modes. Note that this objective has a constant-output minimiser, which is a testable explanation for the observed divergence | `cirr/tta/note_reg.py:1-42` |

### 3.2 Questions that remain OPEN

| ID | Question |
|---|---|
| Q2 | The causal queue is cleared at every day boundary (`cirr/evaluation/online.py:174`), so the last `horizon_steps` samples of each day never produce an update. What fraction of the stream is that, and is this the intended semantics? |
| Q3 | `ogd_step` calls `model.eval()` before the backward pass (`cirr/training/ogd.py:101`), so BatchNorm uses running statistics and does not update them during adaptation. Is that deliberate? It determines whether the "full" action can shift the encoder's normalization at all. |
| Q4 | Which checkpoint should the chapter deploy? `scripts/compare_checkpoints.py` exists precisely to answer this and has apparently never been run for the record. |
| Q5 | Are the 19 missing cells of `tab:ch4-solar-benchmark` recoverable from a machine that still holds them, or must they be re-run? |
| Q6 | Where are the Modena and week-scale streams behind `tab:ch4-solar-metrics`? No dataset in `data/` matches them: the prepared sites run 512-851 samples/day, the table implies 1400-1600. |
| Q7 | Poznań `periods.pickle` holds 215 days, the chapter states 204. Which is the scored set? |
| Q8 | Do the sliding-window and RLS comparators share the trainable scope of the conditional policy's head action, or is the comparison confounded by scope? |
| Q9 | Is a Raspberry Pi 5 available for this phase, and is the inline energy meter of `sec:ch4-heterogeneous-evaluation` still instrumented? |
| Q10 | What is the wall-clock cost of one full-year single-site adaptation run on the RTX 4090? Estimate from existing logs. |

### 3.3 Gap table format

One row per planned item.

| Plan item | Status | Evidence (file:line) | Proposed change | Effort (S under 2 h, M under 1 day, L more) | Risk or dependency |
|---|---|---|---|---|---|

Status values: **exists**, **modify**, **new**, **blocked** (needs a user decision), **out of scope**.

### 3.4 Phase 0 output

Write `docs/CH4_PLAN_VALIDATION.md` with: the traced call chain for one conditional run,
re-verification of R1 to R12, answers or OPEN marks for Q1 to Q10, the gap table, and the
decisions the user must make.

### Checkpoint A

Stop. Present the open questions, the blocked items and your proposed adjustments. Wait for approval.

## 4. Phase 1. Minimal fixes before any new run

Invoke `/ponytail full` first. Add a switch only where a control, an ablation or a
regression check needs one.

| ID | Fix | Acceptance criterion |
|---|---|---|
| F1 | Apply `filter_lit_days` in the Chapter 4 evaluation path, once, where `load_eval_days` builds the day index map | No scored sample has clear-sky GHI below `--min-ghi-clear`; default 0.001 kW/m²; `0` reproduces the old population |
| F2 | Load one source checkpoint shared with Chapter 5, decided by Q4; log its path and SHA-256 in every run | Every `config.yaml` under `results/ch4/` carries the same checkpoint hash as the Chapter 5 campaign, or the deviation is recorded with a reason |
| F3 | Replace the free-text `*_results.txt` with the structured schema of §5.4 | Every file and column in §5.4 is written by every method |
| F4 | State the pooling once. Report pooled metrics, and emit day-weighted alongside so the gap is visible | `summary.json` carries both, and their difference |
| F5 | Trigger on the clamped forecast, matching `eq:ch4-abs-error`, and keep a sentinel for the unclamped alternative | A test asserts `delayed_abs_error` equals `absolute_error`, and `scope_changed_if_raw` totals zero |
| F6 | Make the two actions of Section 4.3 real: `head_only` is the head, `head_and_encoder` is the whole network | A test asserts `head_and_encoder` trains every parameter and `head_only` trains only the head |
| F7 | *Deferred by author decision (2026-09-22).* Trigger controls are not implemented. The frozen reference comes from `scripts/evaluate.py --method none`, which runs through the same campaign contract | — |
| F8 | Log the action mix per day and per sample in the new schema, split by lit and dark | `events.csv` reproduces the R1 and R9 numbers for the legacy population |
| F9 | Add a retention pass: after the campaign, score the adapted model on the held-out Folsom source stream and on the first 30 days of the deployment stream | `retention.json` per run with frozen-model and adapted-model metrics on both |
| F10 | Log retained buffer bytes and peak process memory in the campaign runs, not only in the edge benchmark | `summary.json` carries `buffer_bytes` and `peak_rss_mb` |
| F11 | Make runs deterministic per seed, and seed MEMO, NOTE and few-shot explicitly | Same seed twice gives identical `metrics_daily.csv`; different seeds differ for the stochastic methods |
| F12 | Single-source the comparators' trainable scope with the policy's head action, per Q8 | A test prints the trainable parameter count per method and the head-scope counts are equal |

### Checkpoint B

Stop. Show the diff summary per fix, the trainable parameter counts per method and per action,
and any fix you skipped with the reason. Wait for approval.

## 5. Phase 2. Smoke benchmark on a 7-day subset

Purpose: prove every configuration runs, the logs are complete and the controls behave, before
spending GPU-hours.

### 5.1 Window selection

**15 to 17 June**, recorded in `configs/ch4_smoke_window.yaml`: the first 3 days of the 7-day
candidate below, cut by author decision to keep the smoke short (7,873 samples). Keep the manifest; the device
measurements replay the same window.

Selected by measurement rather than by the Chapter 5 rule, which has no solution here. Coverage
is lit samples present over lit minutes available in the clear-sky table. No 7-day window in
March to May clears 80 % at all three sites — the best, 9 to 15 May, reaches 61 % because of
Karpicko, which additionally has no data at all for 1 to 20 April. Fourteen strictly consecutive
windows clear 80 %; 15 to 21 June is the one that does so with the largest population (0.829
worst single day, at Karpicko; 18,399 samples across the three sites), giving every site its
longest days and enough
mixed-regime error to exercise the full-model action. The March-to-May preference was inherited
from the federated plan and does not survive contact with these sites. Folsom caps near 0.98
because its camera records 06-20 h rather than full daylight, so 1.0 is unreachable anywhere.

Cold-season alternative: 21 to 27 January, 0.918 worst day but 10,593 samples over 7 days.
Measured throughput is 13.5 ms/sample frozen, 16.1 NOTE and 23.6 for the conditional policy.

**Few-shot 7d cannot run on a 3-day window.** `evaluate_fewshot` fits on the first
`fewshot_days` days and predicts on the remainder, so a 3-day window leaves it nothing to
predict. It runs on 15 to 24 June for the executability check only, recorded in the manifest as
`fewshot_d7_exception`; its metrics are not comparable with the other configurations.

### 5.2 Configurations to smoke-test

| Group | Configurations | Entry point |
|---|---|---|
| References | Static (frozen), Smart Persistence | `scripts/evaluate.py --method none`; Smart Persistence is scored inside every run |
| Comparators | MEMO, NOTE, few-shot 1d, few-shot 7d | `scripts/evaluate.py --method memo\|note\|fewshot` |
| Comparators | RLS, sliding window | `scripts/related_works/evaluate_sm_rls.py`, `evaluate_sliding_window.py` |
| Proposed | conditional at T1 = 10, T2 = 100 W/m² | `scripts/related_works/evaluate_lagged_ogd.py` |
| Regression | conditional with `--min-ghi-clear 0` and `--checkpoint checkpoints/epoch_010.ckpt` | reproduces the published population, but **not** the published high-error action (R13) |

### 5.3 Logging schema

```text
results/ch4/<tier>/<site>/<method>/seed<k>/
  config.yaml        resolved config, git commit, checkpoint path and SHA-256
  metrics_daily.csv  date, n_samples, n_dropped_dark, mae, rmse, wape, smape, sp_rmse
  events.csv         date, considered, none, head, whole, lit_none, lit_head, lit_whole,
                     dark_none, dark_head, dark_whole, queue_flushed, scope_changed_if_raw
  predictions.csv    timestamp, prediction, target, absolute_error, delayed_abs_error,
                     delayed_abs_error_raw, adaptation_scope, scope_if_raw, ghi_clear
  retention.json     source-stream and early-segment metrics, frozen vs adapted
  summary.json       pooled and day-weighted MAE/RMSE/WAPE/SMAPE/FS, action totals,
                     buffer_bytes, peak_rss_mb, wall_time_s, gpu_name
```

### 5.4 Smoke acceptance checks

| ID | Check | Pass rule |
|---|---|---|
| S1 | Completion | Every configuration in §5.2 finishes without error |
| S2 | Sanity | No NaN or infinite values; daily MAE below 300 W/m² |
| S3 | Determinism | Seed 0 twice gives identical `metrics_daily.csv`; seed 1 differs for MEMO, NOTE and few-shot |
| S4 | Daylight filter | Zero scored samples with clear-sky GHI below the threshold; `n_dropped_dark` matches R7 shares on the full stream |
| S5 | Action accounting across methods | The conditional policy's `whole` count equals its full-model updates; MEMO and NOTE report one `whole` event per adaptation, matching `adapt_events` in `config.yaml` |
| S6 | Action accounting | `events.csv` totals equal the per-sample `adaptation_scope` counts in `predictions.csv`, and `scope_changed_if_raw` is zero at every site (a nonzero value means the model predicted negative irradiance) |
| S7 | Static agreement | `scripts/evaluate.py --method none` and any zero-update path agree to 1e-8 on the same window |
| S8 | Smart Persistence | Scored on exactly the model population, cross-checked against `compute_sp_metrics.py` conventions |
| S9 | Regression | `--min-ghi-clear 0` with the legacy checkpoint reproduces the published population size (226,114 at Karpicko). Metrics will **not** match, because the published high-error action was a head update (R13); record the gap rather than chasing it |
| S10 | Logs | Every file and column of §5.3 exists for every configuration |
| S11 | Runtime | Wall time recorded and extrapolated to a full campaign; warn above 2 h per run |

### 5.5 Phase 2 output

`docs/CH4_SMOKE_REPORT.md` with the window, a pass/fail table for S1 to S11, extrapolated
GPU-hours per tier, and any configuration needing attention.

### Checkpoint C

Stop. Launch no full campaign until the user approves the smoke report.

## 6. Phase 3. Full campaigns

Sites: Folsom 2016 (363 days), Karpicko (301), Poznań (215 or 204, per Q7). Expect 1 to 2
GPU-hours per site-method run.

### Tier 1 (must, 36 runs, about 33 GPU-hours)

Campaign populations after the daylight filter: Folsom 362 days / 245,538 lit samples, Karpicko
298 / 179,858, Poznań 213 / 123,523.

| Method | Seeds | Why that many |
|---|---|---|
| Static, SM-RLS, sliding window, conditional | 1 | No stochastic operation anywhere in the path |
| NOTE | 1 | Verified: seeds 0 and 7 give bit-identical daily metrics |
| MEMO | 1 | Author decision, 2026-09-23. Three seeds would have added about 20 h for the method that tracks the frozen baseline most closely |
| Few-shot 1d, few-shot 7d | 3 | The train/val split and the shuffle genuinely move results |

That is 12 runs per site. **Seed count is not a proxy for repeatability here.** For the six
single-seed methods a repeat is bit-identical, so between-run variance is zero by construction and
a single run *is* the population value. What remains uncharacterised is sensitivity to the source
checkpoint and to the data order, which is fixed by time — the chapter's "single-run results"
limitation should be restated in those terms rather than as unmeasured noise.

Estimates come from the per-method rates measured in the smoke (`docs/CH4_SMOKE_REPORT.md` §7)
applied to the full campaign populations, plus 10 min per conditional run for the retention pass.

| | GPU-hours |
|---|---|
| MEMO, three sites | 7.1 |
| Few-shot, both variants, three seeds, three sites | 11.1 |
| Conditional, three sites (including retention) | 3.6 |
| Everything else | 10.9 |
| **Total** | **32.7** |

**Four-server split.** `scripts/bash_scripts/run_ch4_tier1.sh <1|2|3|4>` owns a disjoint slice,
balanced longest-run-first: 8.4 h / 8.0 h / 8.4 h / 8.0 h, so the tier completes in **about 8.5
hours wall clock**. Runs are independent and write to disjoint folders, so the servers share
nothing but the repository and `data/`. Any run whose `summary.json` exists is skipped, so a slice
can be restarted or moved to another machine without losing work.

### Checkpoint D

Stop after Tier 1. Run the Phase 4 analysis on Tier 1 only, report the H4.1 verdict (H4.2 is
deferred, §2.2), and wait for approval before Tier 2.

### Tier 2 (sensitivity)

| Experiment | Configurations | Seeds | Runs |
|---|---|---|---|
| Threshold sensitivity | T1 ∈ {0, 5, 10, 20, 40}, T2 ∈ {50, 100, 200} W/m² | 1 | 45 |
| Horizon | H ∈ {5, 15, 30} for conditional and Static, on the campaign streams rather than the missing week-scale ones | 1 | 18 |
| Retention | frozen and adapted models scored on the Folsom source stream after each campaign | — | folded into F9 |

### Tier 3 (device, gated on Q9)

Re-run `scripts/edge_adaptation_benchmark.py` on the Raspberry Pi 5 over the §5.1 window with
the measured action mix, not an unspecified 640-sample slice, and record energy per day
alongside latency. This is the only measurement that can settle C4.3.

## 7. Phase 4. Analysis

Invoke `/ponytail full`. Put all analysis in one module that reads only `results/ch4/`, and
import the estimators from `cirr/fl/analysis.py` rather than restating them.

### 7.1 Pre-registered settings (do not change without asking)

| Setting | Value |
|---|---|
| Scored population | Clear-sky GHI ≥ 0.001 kW/m², applied once at the day index map |
| Primary metric | MAE on the lit population |
| Secondary | Forecast skill against Smart Persistence, RMSE, WAPE, SMAPE |
| Pooling | Pooled over samples; day-weighted reported alongside with the gap |
| Confidence intervals | Moving-block bootstrap over days, block length 7, 2 000 resamples, paired by date |
| Seed handling | Average daily errors across seeds before bootstrapping; report seed standard deviation separately |
| Equivalence (H4.1) | Two one-sided tests at α = 0.05 against sliding-window adaptation, margin ±5 % relative MAE, reported as a 90 % CI |
| Pairwise comparisons | Diebold-Mariano on daily loss differentials, Newey-West variance at lag 6, Holm correction across sites and methods |
| Update cost | Per-site action proportions × measured per-action latency and energy, reported per deployed day |
| Retention (H4.4) | Adapted-model MAE on the source stream relative to the frozen model, per site |

### 7.2 Outputs

| Output | Content |
|---|---|
| `tables/ch4_main.csv` and `.tex` | Replacement for `tab:ch4-solar-benchmark` on the lit population, with CIs, Smart Persistence and Static rows |
| `tables/ch4_equivalence.csv` | Relative MAE difference to sliding window with 90 % CI and TOST verdict per site |
| `tables/ch4_dm.csv` | Diebold-Mariano statistics with Holm-adjusted p-values |
| `tables/ch4_controls.csv` | Conditional against always / never / random triggers |
| `tables/ch4_actions.csv` | Per-site action proportions, split lit and dark, with the implied campaign update cost |
| `tables/ch4_thresholds.csv` | Tier 2 threshold sweep |
| `tables/ch4_retention.csv` | Source-stream and early-segment retention per site |
| `tables/ch4_population_delta.csv` | Every Tier 1 cell under the legacy and filtered populations, so the chapter can state what the filter changed |
| `figures/` | Daily error relative to Static, monthly, NOTE in its own panel (closes REVISION NOTE TW-08); action-mix over the calendar; threshold dose-response |
| `docs/CH4_ANALYSIS_REPORT.md` | Verdicts for H4.1 to H4.4 with the supporting numbers and every deviation from this plan |

### 7.3 Verdict rules

1. **H4.1 supported** if every site's 90 % CI lies inside ±5 %; **not supported** if any interval crosses the margin.
2. **H4.2 is not tested in this campaign.** Trigger controls were deferred, so the section may not claim the error threshold is doing work; it may only report what the policy achieves at the update rate it selects.
3. **H4.3 supported** only if the measured action mix, not a 640-sample slice, gives the lower campaign cost at equal or better MAE.
4. **H4.4 supported** if source-stream degradation is at most 10 % at every site; otherwise report the forgetting cost as a finding, not a limitation.

## 8. Chapter edits this campaign licenses

Independent of the runs, Phase 0 alone already licenses these corrections. Make them at
Checkpoint A, not at the end.

1. Delete the claim that action proportions are unavailable (`sec:ch4-adaptation-protocol`, `sec:ch4-system-factorization`, `sec:ch4-conditional-adaptation-evaluation`) and report R1.
2. State that every published result came from a two-action policy, not the three-action policy the chapter formalizes (R13). The high-error branch of `eq:ch4-update-trigger` and the "update full model" line of `alg:ch4-conditional-adaptation` were never executed as written, so the campaign never paid the three-inference cost that the discussion of `tab:ch4-on-device-performance-results` attributes to a full update. The code now implements the two actions as described, which means the Tier 1 numbers replace rather than refine the published ones.
3. Replace "the buffered tensors and timed update path remain unresolved" with R2 and R3: the image is retained, features are recomputed, and the buffer is ~2.9 MB.
4. Replace the MEMO and NOTE "not documented" hedges with R11 and R12, and state the constant-output minimiser as the candidate explanation for NOTE's divergence.
5. Replace the WAPE/MAE reconciliation note with R5 and R6, which are definitions rather than open questions.
6. State the night-sample defect (R7 to R9) explicitly as the reason the tables are provisional, with the Karpicko +20.4 % figure as its measured size, rather than as a suspicion about sample counts.
7. Record that Chapters 4 and 5 use different source checkpoints (R10).

## 9. Blocked items

| Item | Why blocked | Needed decision |
|---|---|---|
| `tab:ch4-solar-metrics` (Modena, week-scale, H ∈ {5,15,30}) | No matching dataset in `data/` (Q6) | Supply the streams, or re-scope the horizon study onto the campaign streams as in Tier 2 |
| The Modena sample-count reconciliation | Same | Same |
| Energy per deployed day (C4.3) | Needs the Pi 5 and the inline meter (Q9) | Confirm hardware availability |
| 19 missing cells of `tab:ch4-solar-benchmark` | Artifacts not in the repository (Q5) | Recover them, or accept the Tier 1 re-run as the record |
| H4.2 (is the trigger active?) | Controls deferred by author decision | Re-open F7 when the chapter needs the mechanism claim |

## 10. Final deliverables checklist

- [ ] `docs/CH4_PLAN_VALIDATION.md` (Phase 0)
- [ ] Chapter edits of §8 applied at Checkpoint A
- [ ] F1 to F12 committed on `feature/adaptation`, with skipped items listed
- [ ] `configs/ch4_smoke_window.yaml` and `docs/CH4_SMOKE_REPORT.md` (Phase 2)
- [ ] Tier 1 and 2 results under `results/ch4/` with an empty `failures.log`
- [ ] Tables, figures and `docs/CH4_ANALYSIS_REPORT.md` (Phase 4)
- [ ] `docs/CH4_REPRODUCE.md` with the exact commands for the smoke test and each tier
