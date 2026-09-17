# Chapter 5 server benchmark plan for Claude Code

**How to use.** Save this file in the repository as `docs/ch5_server_benchmark_plan.md`, open Claude Code in VS Code at the repository root, and send the message below. You can also paste everything under the line directly into the chat.

```text
Read docs/ch5_server_benchmark_plan.md and execute Phase 0. Stop at Checkpoint A.
```

---

# Instructions for Claude Code

## 1. Role and ground rules

You are extending the experiment code behind Chapter 5 of a doctoral dissertation on regime-stratified prototype exchange for multi-site solar irradiance forecasting. Your job is to validate this plan against the code that exists, make the smallest changes needed, run a 7-day smoke benchmark, and then run and analyze the full-year benchmarks on the server GPU.

Follow these rules in every phase.

1. **Use `/ponytail` for all code work.** Invoke `/ponytail full` before you write, modify, refactor, or review any code in Phases 1 to 4, and invoke it again after any context compaction. Reading code and running existing commands do not need it. The reports this plan asks for (`PLAN_VALIDATION.md`, `SMOKE_REPORT.md`, `ANALYSIS_REPORT.md`) are explicitly requested, so write them in full.
2. **Stop at every checkpoint.** At Checkpoints A, B, C, and D, summarize what you did and wait for the user's approval before continuing.
3. **Do not invent values.** If the code does not answer a question in this plan, list it as an open question. Never guess a loss form, a hyperparameter, or a data filter.
4. **Protect existing results.** Work on a new branch `ch5-benchmarks`, never overwrite or delete existing result folders, and commit at the end of each phase with a message that names the phase.
5. **No new dependencies without asking.** First check what is installed (for example `numpy`, `pandas`, `scipy`, `pvlib`, `torch`, `statsmodels`) and reuse it.
6. **Server GPU only.** This plan covers task metrics and event counts computed on the server. On-device latency, energy, and memory are measured later on a Raspberry Pi 5 with a separate plan, so do not add device measurement code here.
7. **Reuse before adding.** Extend the existing training loop, baselines, metrics, and configuration system. Add a new file only when no existing module is the right place.

## 2. Scientific context (fixed by the dissertation)

### 2.1 Hypotheses the server benchmarks must test

| ID | Hypothesis | Pre-registered success criterion |
|---|---|---|
| H5.1 | Prototype exchange is safe under structured heterogeneity | At every site, the 90% confidence interval of the relative MAE difference to Isolated lies inside ±2% |
| H5.2 | The exchange mechanism is active, not inert | The shuffled-prototype control changes MAE beyond seed variability, and the proposed method differs from the own-prototype control |
| H5.3 | Collaboration pays where local data are scarce | In cold-start or late-join scenarios, fewer days to reach the accuracy threshold, or lower cumulative excess error over the first 30 days, with a 95% confidence interval that excludes zero |
| H5.4 | The cost advantage holds on realistic links | Evaluated later from device unit costs; the server runs only log the event counts it needs (Section 5.4) |

### 2.2 Current setup described in the thesis (verify each item against the code in Phase 0)

| Item | Thesis description |
|---|---|
| Task | Single-step 15-minute-ahead global horizontal irradiance regression |
| Model | ImageNet-pretrained ResNet50 encoder, pooled 2048-d features concatenated with four irradiance values ($I_t$, $I_{t-5}$, $I_{t-10}$, $I_{t-15}$), head 2052 → 256 (ReLU) → 1 |
| Source model | Folsom 2014 to 2015, AdamW lr 3e-4 halved after 5 epochs without improvement, batch 128, at most 30 epochs, patience 10, 20% of days held out for validation |
| Clients | Folsom 2016, Poznań (2024-10-18 to 2025-07-01, 204 days), Karpicko (2024-11-01 to 2025-10-31, 301 days) |
| Calendar | Normalized calendar year, dates ordered by month and day with the year ignored |
| Client training | AdamW lr 1e-5, weight decay 1e-4, gradient clipping 1.0, batch 32, one local epoch per day |
| Schedule | 30-day isolated warm-up, then synchronization every 7 days |
| Regimes | Proxy $k_c$ = target irradiance / 1000 W/m², classes overcast < 0.15 ≤ transient < 0.50 ≤ clear, so C = 3 |
| Prototype weighting | Attention over peer prototypes with temperature τ = 0.5 |
| References | Static (no adaptation), Isolated (local adaptation, no exchange), Smart Persistence (Ineichen–Perez clear-sky index) |
| Federated baselines | FedAvg, FedProx, FedBN, FedPer, FedEMA |
| Metrics | MAE, RMSE, WAPE, SMAPE, forecast skill FS = 1 − RMSE_model / RMSE_SmartPersistence |
| Known gaps | Alignment loss form and weight λ undefined in the text; figures show a frozen encoder while the equations index client-specific encoder weights; payload bytes inconsistent with 32-bit values; local training time differs by 4.4 s between FedAvg and prototype runs; a separate 36.3 s prototype extraction pass exists |

## 3. Phase 0. Discovery and plan validation (read-only)

Do not modify any file in this phase. If plan mode is available, use it.

### 3.1 Tasks

1. Map the repository. List the entry points, the configuration system, the data loaders for each site, the federated simulation loop, each baseline, the prototype construction and attention code, the metric code, the logging code, and where results are written.
2. Trace one run of the proposed method end to end, from data loading to the metrics file, and write the call chain in the report.
3. Answer every question in Section 3.2 with a file path and line numbers as evidence. If the code does not answer a question, mark it **OPEN**.
4. Build the gap table in Section 3.3 for every item in Sections 4 to 7.
5. Estimate the runtime of one full-year run from existing logs, if any exist.

### 3.2 Questions to answer from the code

| ID | Question |
|---|---|
| Q1 | What is the exact alignment loss (functional form, weight λ, and the tensors it compares)? |
| Q2 | Which parameters receive gradients during local training for Isolated, each federated baseline, and the proposed method? Is the encoder frozen or trained in each? |
| Q3 | On which layer are prototypes computed, and what is their dimension? |
| Q4 | Is the global prototype held fixed between rounds? Does the attention sum include the client's own prototype? |
| Q5 | How is the regime label computed, and is the Ineichen–Perez clear-sky irradiance already available in the data pipeline? |
| Q6 | Is any daylight or solar-zenith filter applied in training, evaluation, or both? |
| Q7 | Which source checkpoint is loaded, and is it the same one Chapter 4 uses? |
| Q8 | How is the normalized calendar built, and how are missing days and gaps handled? |
| Q9 | How are payload bytes computed (tensor size times dtype, serialized size, or another method)? Uplink, downlink, or both? |
| Q10 | Why does prototype extraction need a separate pass instead of reusing embeddings from the daily forward pass? |
| Q11 | What explains the 4.4 s difference in local training time between FedAvg and prototype runs? |
| Q12 | Which seeds are set, and are cuDNN and DataLoader settings deterministic? |
| Q13 | Is client participation (dropout) already supported? |
| Q14 | Is a mid-run client join (cold start or late join) already supported? |
| Q15 | Where is Smart Persistence computed, and is its RMSE logged per site? |

### 3.3 Gap table format

For each planned item, write one row.

| Plan item | Status | Evidence (file:line) | Proposed change | Effort (S under 2 h, M under 1 day, L more) | Risk or dependency |
|---|---|---|---|---|---|

Status values are **exists**, **modify**, **new**, **blocked** (needs a user decision), and **out of scope**.

### 3.4 Phase 0 output

Write `docs/PLAN_VALIDATION.md` with the repository map, the traced call chain, answers Q1 to Q15, the gap table, a list of plan items that do not fit the current software, and a proposed adjustment for each. Include a short list of decisions the user must make.

### Checkpoint A

Stop. Present the open questions, the blocked items, and your proposed adjustments. Wait for approval.

## 4. Phase 1. Minimal fixes required before any new run

Invoke `/ponytail full` before starting. Each fix must stay switchable only where a switch is needed for a control, an ablation, or a regression check. Do not add switches for values that never change.

| ID | Fix | Acceptance criterion |
|---|---|---|
| F1 | Implement or document the alignment loss from Q1 and expose λ as a parameter | Loss form and λ written in `docs/PLAN_VALIDATION.md`; λ read from config |
| F2 | Make the trainable scope identical for Isolated, all federated baselines, and the proposed method; if the encoder is frozen, compute prototypes on a trainable layer (for example the 256-unit hidden layer) | A test prints the trainable parameter count per method, and the counts are equal |
| F3 | Add the clear-sky regime label $k^{cs}_c$ = GHI / GHI_cs (Ineichen–Perez) next to the old proxy | Config key `regime_label: proxy \| clear_sky`, default `clear_sky` |
| F4 | Apply one daylight filter, solar zenith angle below 85°, in training and evaluation, shared with Chapter 4 | No evaluated sample has zenith of 85° or more; filter defined once |
| F5 | Load one shared source checkpoint for Chapters 4 and 5 | Checkpoint path and hash logged in every run |
| F6 | Add prototype controls `align_target: global \| own \| shuffled \| random` | `own` sends zero bytes; `shuffled` permutes regime labels of received prototypes each round; `random` uses Gaussian vectors with matched norm |
| F7 | Add ablation switches `attention: on \| uniform` and `stratify: on \| single` | Both switches change the round log as expected |
| F8 | Add participation schedules `participation: full \| dropout:p \| outage:client:start:days` | Absent clients neither send nor receive in their rounds, and the log shows it |
| F9 | Add client join `join_day: <int>` per client, with `local_history_days: <int>` for cold start | A joining client has no training data before its join day except the stated history |
| F10 | Make runs deterministic per seed | Same seed twice gives identical daily metrics; different seeds differ |
| F11 | Add the logging schema in Section 5.4 | Every file and column in Section 5.4 is written |
| F12 | Compute byte counts from the actual serialized payload | Logged bytes equal the length of the serialized buffer for FedAvg and prototypes |
| F13 | Reuse embeddings from the daily forward pass for prototype extraction, if Q10 shows the separate pass is not required | Extraction adds no second forward pass, or the reason it must is documented |
| F14 | Add the pooled oracle, one model trained daily on the union of all clients' data up to that day | Runs through the same evaluation code as the other methods |
| F15 | Add FedAvg with local fine-tuning and FedProto if they do not exist; add Ditto only if it takes less than a day | Each runs through the same loop and logs |

Suggested determinism settings, to be checked against the existing code.

```python
import os, random, numpy as np, torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
# DataLoader: worker_init_fn seeding and a torch.Generator seeded per run
```

### Checkpoint B

Stop. Show the diff summary per fix, the trainable parameter counts per method, and any fix you skipped with the reason. Wait for approval.

## 5. Phase 2. Smoke benchmark on a 7-day subset

The purpose is to prove that every configuration runs end to end, that the logs are complete, and that the controls behave as designed, before spending GPU-hours.

### 5.1 Window selection

1. In the normalized calendar, choose the first 7 consecutive days, preferring March to May, on which every client has at least 80% of its expected daylight samples.
2. Record the chosen dates per site in `configs/smoke_window.yaml`.
3. Keep this window manifest, since the device measurements will later replay the same window.

### 5.2 Shortened schedule

| Setting | Smoke value | Full-year value |
|---|---|---|
| Warm-up | 2 days | 30 days |
| Synchronization | every 2 days (rounds on days 3, 5, and 7) | every 7 days |
| Seeds | 1, plus a repeat of seed 0 for the determinism check | 5 or 3 (Section 6) |
| Cold-start join | client joins on day 3 with 0 days of history | Section 6 |

### 5.3 Configurations to smoke-test

Run each configuration once with seed 0. Static and Smart Persistence need no training but must pass through the evaluation code.

| Group | Configurations |
|---|---|
| References | Static, Smart Persistence, Isolated, pooled oracle |
| Federated baselines | FedAvg, FedProx, FedBN, FedPer, FedEMA, FedAvg with fine-tuning, FedProto (and Ditto if added) |
| Proposed | proposed with `regime_label: clear_sky`, proposed with `regime_label: proxy` |
| Controls | `align_target` own, shuffled, random |
| Ablations | `attention: uniform`; `stratify: single`; both |
| Participation | proposed with `dropout:0.5`; proposed with a 2-day outage of one client |
| Joins | proposed and Isolated with a cold-start client on day 3 |
| Regression | Isolated and proposed with the old settings (proxy label, no daylight filter, old checkpoint) |

### 5.4 Logging schema (checked in the smoke run)

```text
results/ch5/<tier>/<experiment>/<method>/seed<k>/
  config.yaml          full resolved config, git commit, checkpoint hash
  metrics_daily.csv    date, client, n_samples, mae, rmse, wape, smape, sp_rmse
  metrics_regime.csv   date, client, regime, n_samples, mae, rmse
  events.csv           date, client, samples_after_filter, train_steps, extraction_calls,
                       participated, bytes_up, bytes_down, payload_dtype
  rounds.csv           round, date, client, align_loss_mean, update_l2_norm, prototype_norms
  attention.npy        rounds x clients x clients x regimes
  summary.json         per-client MAE, RMSE, WAPE, SMAPE, FS, wall_time_s, gpu_name
```

### 5.5 Smoke acceptance checks

| ID | Check | Pass rule |
|---|---|---|
| S1 | Completion | Every configuration in 5.3 finishes without error |
| S2 | Sanity | No NaN or infinite values; daily MAE below 300 W/m² |
| S3 | Determinism | Seed 0 run twice gives identical `metrics_daily.csv`; seed 1 differs |
| S4 | Daylight filter | Zero evaluated samples with zenith of 85° or more |
| S5 | Regime labels | Class counts are reported per client, with a warning for any class with zero samples |
| S6 | Controls | `own` logs zero bytes; `shuffled` and `random` change the alignment target (assert on the tensors in round 1) |
| S7 | Byte accounting | Logged bytes equal the serialized buffer length for FedAvg and prototypes |
| S8 | Participation and joins | Absent or not-yet-joined clients show `participated = 0` and no bytes in those rounds |
| S9 | Logs | Every file and column in 5.4 exists for every configuration |
| S10 | Regression | Old-settings Isolated and proposed runs reproduce the pre-change code on the same 7 days, with daily MAE within 1e-4 W/m² (or explain the difference) |
| S11 | Runtime | Wall time per configuration recorded and extrapolated to a full year (warn if above 2 h) |

### 5.6 Phase 2 output

Write `docs/SMOKE_REPORT.md` with the chosen window, a pass or fail table for S1 to S11, the extrapolated GPU-hours per tier from Section 6, and any configuration that needs attention.

### Checkpoint C

Stop. Do not launch any full-year run until the user approves the smoke report.

## 6. Phase 3. Full-year benchmarks

Invoke `/ponytail full` before writing the runner. Use the existing launcher if there is one. Otherwise, write one short script that reads a list of configurations, skips any run whose `summary.json` exists, runs N jobs in parallel on the GPU, and writes failures to `results/ch5/failures.log`.

Expected cost is 1 to 2 GPU-hours per full-year run.

### Tier 1 (must, 80 runs, about 80 to 160 GPU-hours)

| Experiment | Configurations | Seeds | Runs |
|---|---|---|---|
| Main comparison | Isolated, FedAvg, FedProx, FedBN, FedPer, FedEMA, proposed | 5 | 35 |
| E1 mechanism controls | proposed with `align_target` own, shuffled, random | 5 | 15 |
| E8 component ablation | `attention: uniform`; `stratify: single`; both | 5 | 15 |
| E9 participation | proposed with `dropout:0.25`, `dropout:0.5`, one 21-day outage per client in turn | 5 | 15 |

### Checkpoint D

Stop after Tier 1 and run the Phase 4 analysis on Tier 1 only. Report the H5.1 and H5.2 verdicts and wait for approval before Tier 2.

### Tier 2 (upside, about 53 run-equivalents, about 53 to 106 GPU-hours)

| Experiment | Configurations | Seeds | Runs |
|---|---|---|---|
| E3 cold start | Poznań and Karpicko each as the new client, with 0, 1, 3, 7 days of history, for Isolated, FedAvg, proposed; 60-day runs | 3 | 72 short, about 12 full-run equivalents |
| E4 late join | Poznań and Karpicko each joining after 90 days, Isolated and proposed | 3 | 12 |
| E6 pooled oracle | pooled training on all clients | 5 | 5 |
| E7 extra baselines | FedAvg with fine-tuning, FedProto, Ditto if implemented | 5 | 15 |
| Concurrent scenario | Poznań and Karpicko live over their calendar overlap (2024-11-01 to 2025-07-01), Folsom as a fixed archival prototype set; Isolated, FedAvg, proposed | 3 | 9 |

### Tier 3 (sensitivity, 42 runs, about 42 to 84 GPU-hours)

| Experiment | Configurations | Seeds | Runs |
|---|---|---|---|
| Regime label | proposed with the old proxy label | 3 | 3 |
| Number of regimes | C = 2, C = 4 (cut points from clear-sky index terciles or halves, documented) | 3 | 6 |
| Temperature | τ = 0.1, τ = 1.0 | 3 | 6 |
| Alignment weight | λ at 0.25× and 4× the default | 3 | 6 |
| Schedule | warm-up and sync of (0, 1), (0, 7), (7, 1), (7, 7), (30, 1) days | 3 | 15 |
| Parameter averaging under dropout | FedAvg with `dropout:0.25` and `dropout:0.5` | 3 | 6 |

## 7. Phase 4. Analysis

Invoke `/ponytail full` before writing analysis code. Put all analysis in one script or notebook that reads only `results/ch5/`.

### 7.1 Pre-registered analysis settings (do not change without asking)

| Setting | Value |
|---|---|
| Primary metric for method comparisons | Forecast skill against Smart Persistence |
| Metric for absolute error and H5.1 | MAE |
| Equivalence margin (H5.1) | ±2% relative MAE difference to Isolated, per site |
| Confidence intervals | Moving-block bootstrap over days, block length 7 days, 2,000 resamples, paired by date |
| Seed handling | Average daily errors across seeds before bootstrapping; report seed standard deviation separately |
| Equivalence test | Two one-sided tests at α = 0.05, reported as a 90% CI |
| Pairwise comparisons | Diebold–Mariano on daily loss differentials with a Newey–West variance (lag 6), Holm correction across sites and methods |
| Cold-start threshold | First day on which the 7-day rolling MAE of the new client is at most 1.05 × the rolling MAE of the warm Isolated run over the same dates |
| Cumulative excess error | Sum over days 1 to 30 of (new-client daily MAE minus warm Isolated daily MAE), with a block-bootstrap CI |
| Daylight filter | Solar zenith below 85° |

### 7.2 Outputs

| Output | Content |
|---|---|
| `tables/ch5_main.csv` and `.tex` | Table 5.3 with MAE, RMSE, WAPE, SMAPE, FS, and 95% CIs per site; Smart Persistence and Static rows included |
| `tables/ch5_equivalence.csv` | Relative MAE difference to Isolated with 90% CI and TOST verdict per method and site |
| `tables/ch5_dm.csv` | Diebold–Mariano statistics and Holm-adjusted p-values |
| `tables/ch5_controls.csv` | E1 controls against proposed and Isolated |
| `tables/ch5_ablation.csv` and `.tex` | Table 5.6 |
| `tables/ch5_participation.csv` and `.tex` | Table 5.5 |
| `tables/ch5_regime.csv` | Per-regime MAE and class counts per site |
| `tables/ch5_coldstart.csv` | Days to threshold and 30-day cumulative excess error with CIs |
| `tables/ch5_sensitivity.csv` | Tier 3 results |
| `tables/ch5_event_counts.csv` | Per client and year totals of samples, train steps, extraction calls, rounds participated, bytes up and down, for the device cost model |
| `figures/` | Cold-start time-to-accuracy curves; attention heatmap over rounds; alignment-loss and update-norm curves; per-regime error bars |
| `docs/ANALYSIS_REPORT.md` | Verdict for H5.1 to H5.3 against the pre-registered criteria, with the numbers that support each verdict and any deviation from this plan |

### 7.3 Verdict rules for the report

1. **H5.1 supported** if every site's 90% CI for the proposed method lies inside ±2%, and **not supported** if any site's interval crosses the margin.
2. **H5.2 supported** if the shuffled control differs from the proposed method with a Holm-adjusted p below 0.05, and the own-prototype control is not equivalent to the proposed method. If the proposed method is equivalent to the own-prototype control, report the do-no-harm result as trivial.
3. **H5.3 supported** if at least one site shows a cold-start or late-join benefit whose 95% CI excludes zero, and **not supported** otherwise.

## 8. Final deliverables checklist

- [ ] `docs/PLAN_VALIDATION.md` (Phase 0)
- [ ] Code changes F1 to F15 committed on `ch5-benchmarks`, with skipped items listed
- [ ] `configs/smoke_window.yaml` and `docs/SMOKE_REPORT.md` (Phase 2)
- [ ] Tier 1, 2, and 3 results under `results/ch5/` with no entries in `failures.log`
- [ ] Tables, figures, and `docs/ANALYSIS_REPORT.md` (Phase 4)
- [ ] A short `docs/REPRODUCE.md` with the exact commands to rerun the smoke test and each tier
