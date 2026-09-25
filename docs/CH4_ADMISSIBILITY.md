# Chapter 4 — input admissibility: reproduction guide

Everything behind the input-admissibility results of Chapter 4 (Section 4.2 and its
composition with the conditional policy of Section 4.3) is produced by the commands
below, in order. Every table under `tables/ch4_admissibility_*.csv` and every figure
under `figures/ch4/ch4_admissibility_*` comes from them; none is edited by hand.

Environment: `~/.conda/envs/irr`, one CUDA GPU, the prepared datasets under `data/`
and the source checkpoint `checkpoints/resnet50_epoch_015.ckpt` (the model shared
with Chapter 5). Long runs write their own logs; the commands below redirect to
`logs/ch4/admissibility/`.

## Code

| File | Role |
|---|---|
| `cirr/evaluation/admissibility.py` | embeddings, gate I/O, the nominal fit/calibration split, the three synthetic faults, the fault schedule and the `CorruptedStream` wrapper |
| `scripts/fit_anomaly_gate.py` | fit the Isolation Forest on Folsom 2014–2015 and score the 2016 stream |
| `scripts/score_anomaly_gate.py` | score any stream (optionally corrupted) with the fitted gate |
| `scripts/related_works/evaluate_lagged_ogd.py` | the policy; `--anomaly-gate` puts the gate in series (Algorithm 1, line 1), `--corruption-schedule` runs it on a corrupted stream |
| `scripts/admissibility_study.py` | the five analyses: `schedule`, `reliability`, `detection`, `crosssite`, `adaptation` |
| `scripts/plot_ch4_admissibility.py` | all figures, from the tables |
| `tests/test_ch4_admissibility.py` | faults are reproducible and stay in the field of view, labels are never altered, the split never reaches the evaluation year |

## 1. Fit the gate (≈ 10 min)

```bash
python scripts/fit_anomaly_gate.py --out results/ch4/gate/folsom
```

Isolation Forest, 100 trees, on the 2048-d embedding the forecasting head consumes.
Folsom **2014–2015 only**: `train_periods.pickle` also holds 2016, the evaluation
year, and the script refuses to fit on it. Every 5th frame, split by whole days:
62,251 frames (458 days) to fit, 15,466 (115 held-out days) to set the threshold at
their 99th percentile. Writes `iforest.joblib`, `gate.json` (threshold, flag rates)
and `scores.csv` (one score per 2016 frame).

## 2. Score the other streams (≈ 5 min each)

```bash
python scripts/admissibility_study.py schedule          # results/ch4/gate/folsom_corrupted/schedule.csv
python scripts/score_anomaly_gate.py --gate results/ch4/gate/folsom --stream data/folsom_prepared \
    --corruption-schedule results/ch4/gate/folsom_corrupted/schedule.csv --out results/ch4/gate/folsom_corrupted
for site in karpicko poznan; do
  python scripts/score_anomaly_gate.py --gate results/ch4/gate/folsom --stream data/${site}_prepared \
      --out results/ch4/gate/$site
done
```

The schedule (seed 0) puts synthetic faults on 3.35 % of the Folsom 2016 daylight
stream: soiling on 1.99 % (4 episodes of 1–3 whole days), droplets on 0.93 % (20
episodes of 1–3 hours), occlusion on 0.43 % (59 episodes of 5–30 minutes). Other
sites are scored with the Folsom forest and threshold unchanged.

## 3. Policy runs (≈ 1–1.5 h each; run concurrently)

```bash
P=scripts/related_works/evaluate_lagged_ogd.py
C="--dataset-path data/folsom_prepared --start-day 0101 --num-days -1 --seed 0 --no-retention"
R=results/ch4/gate/runs
# gate in series, clean stream
python $P $C --anomaly-gate results/ch4/gate/folsom --t2 1e9 --out-dir $R/folsom/headonly_gated/seed0
python $P $C --anomaly-gate results/ch4/gate/folsom --encoder-blocks 2 --encoder-lr 1e-6 --out-dir $R/folsom/last2_lowlr_gated/seed0
python $P $C --anomaly-gate results/ch4/gate/folsom --out-dir $R/folsom/full_gated/seed0
# corrupted stream, without and with the gate
S="--corruption-schedule results/ch4/gate/folsom_corrupted/schedule.csv"
G="--anomaly-gate results/ch4/gate/folsom_corrupted"
python $P $C $S --t2 1e9 --out-dir $R/folsom_corrupted/headonly/seed0
python $P $C $S $G --t2 1e9 --out-dir $R/folsom_corrupted/headonly_gated/seed0
python $P $C $S --encoder-blocks 2 --encoder-lr 1e-6 --out-dir $R/folsom_corrupted/last2_lowlr/seed0
python $P $C $S $G --encoder-blocks 2 --encoder-lr 1e-6 --out-dir $R/folsom_corrupted/last2_lowlr_gated/seed0
```

The ungated clean-stream references are the Section 4.3 runs already under
`results/ch4/ablation/folsom/` and `results/ch4/tier1/folsom/`.

## 4. Analyses and figures (minutes)

```bash
python scripts/admissibility_study.py reliability   # score vs forecast error, risk-coverage, flagged vs admitted
python scripts/admissibility_study.py detection     # synthetic faults: 3 embedding detectors + 2 model-free baselines
python scripts/admissibility_study.py crosssite     # flag rate per site and month, relation to daily error
python scripts/admissibility_study.py adaptation    # corrupted stream: does gating protect what the policy learns?
python scripts/plot_ch4_admissibility.py            # figures/ch4/ch4_admissibility_*.{pdf,png}
pytest tests/test_ch4_admissibility.py
```

| Output | Content |
|---|---|
| `tables/ch4_admissibility_signals.csv` | Spearman of each label-free signal with the absolute forecast error; rank R² of error on history + clear-sky index, with and without the anomaly score |
| `tables/ch4_admissibility_risk_coverage.csv` | MAE of the kept forecasts as the most suspicious are withheld, per signal and site |
| `tables/ch4_admissibility_flagged.csv` | model and Smart Persistence MAE on flagged vs admitted frames; the hybrid that falls back to Smart Persistence |
| `tables/ch4_admissibility_detection.csv` | per detector and fault: share flagged at the 1 % threshold, AUROC against clean 2016 frames |
| `tables/ch4_admissibility_crosssite*.csv` | flag rate per site, per month and per day, with the frozen model's daily error and the day's adaptation gain |
| `tables/ch4_admissibility_adaptation*.csv` | MAE on fault-free and faulty frames, updates made on faulty frames, and bootstrap contrasts on fault-free frames |

Every detector threshold is the 99th percentile of that detector's scores on the
same held-out 2014–2015 frames, so all are compared at the same nominal
false-alarm rate. Contrasts use the estimators of `cirr/fl/analysis.py`
(7-day moving-block bootstrap, 2,000 resamples; Diebold–Mariano with Newey–West lag 6).

## 5. Local-feature maintenance alert — pre-registered design (fixed 2026-09-24, before any evaluation)

The embedding gate above is blind to sensor faults (Section 4 outputs): it scores the globally
pooled embedding of an encoder trained to ignore anything that does not change irradiance. The
aim of the PP-RAI paper is a working **maintenance alert** — telling the operator the dome needs
attention — so the detector below is designed for local faults. Everything in this section was
fixed and committed before it was scored on any synthetic fault.

**Features.** The forecaster's own `layer2` activations (512 × 16 × 16 at 128 × 128 input, one
cell per 8 × 8-pixel block), taken from the forecasting pass itself; 3 × 3 average pooling for
neighbourhood context; a fixed Gaussian random projection to 128 dimensions (seed 0). Only cells
whose 8 × 8 block lies entirely inside the circular field of view are scored.

**Per-position model.** The camera is fixed, so every cell has its own notion of normal. For each
cell, a memory bank of that cell's features from 2,000 nominal frames drawn from the Isolation
Forest's fit days (Folsom 2014–2015, same day split). A cell's score is its nearest-neighbour
distance to its bank divided by the 95th percentile of the same distance on 1,000 other fit-day
frames. The frame score is the maximum over cells.

**Frame threshold.** The 99th percentile of frame scores on 3,000 frames of the held-out
calibration days: a 1 % nominal false-alarm rate, as for every other detector.

**Maintenance alert.** Causal rolling median of the frame score over the preceding 30 minutes of
the same day. The alert threshold is the 99th percentile of that statistic on 30 whole
calibration days (every frame, so the 30-minute window is real time).

**Pass criteria.**

1. *Frame level*, synthetic probe of Section 4: at the 1 % threshold, at least 50 % of soiling
   frames and at least 50 % of occlusion frames flagged, and more than the dark-pixel baseline
   for each.
2. *Maintenance level*, corrupted 2016 stream: every one of the 4 soiling episodes raises an
   alert within its first 2 hours, and alerts fire on at most 1 % of fault-free frames.

Droplets are reported without a criterion: at 128 × 128 a droplet covers a few pixels, and they
may be below the input resolution. The real 2016 stream is checked qualitatively: the frames it
scores highest are rendered for inspection, and are reported as they come.

### 5.1 Reproduction

```bash
python scripts/fit_local_gate.py            # ≈ 1.5 h; --resume reuses local_gate.npz and scores_clean.csv
python scripts/admissibility_study.py detection --workers 2
python scripts/admissibility_study.py maintenance
python scripts/plot_ch4_admissibility.py    # maintenance + local_top figures
```

Outputs: `results/ch4/gate/local/` (bank, scores, `gate.json`), `tables/ch4_admissibility_maintenance_*`,
`tables/ch4_admissibility_local_top.csv`, and the `local (layer2, per cell)` rows of
`tables/ch4_admissibility_detection.csv`. Thresholds: frame 1.7527, alert 1.6356 (deterministic).

### 5.2 Result against the pre-registered criteria: both FAIL

`tables/ch4_admissibility_maintenance_verdict.json`.

**Criterion 1 (frame level): FAIL.** At the 1 % threshold on the synthetic probe:

| detector | clean | soiling | droplets | occlusion | AUROC soil. | AUROC occl. |
|---|---|---|---|---|---|---|
| local (layer2, per cell) | 1.8 % | **13.3 %** | 2.0 % | **50.5 %** | **0.894** | **0.960** |
| Isolation Forest (embedding) | 1.2 % | 0.9 % | 1.0 % | 1.2 % | 0.690 | 0.623 |
| Mahalanobis (embedding) | 0.6 % | 1.3 % | 0.7 % | 0.6 % | 0.714 | 0.620 |
| dark-pixel fraction | 0.7 % | 0.6 % | 0.6 % | 22.9 % | 0.493 | 0.740 |

Occlusion clears its bar (50.5 % ≥ 50 %, 2.2× the dark-pixel baseline); soiling does not (13.3 %).

**Criterion 2 (maintenance level): FAIL, on both parts.** Soiling episodes alerted within 2 h:
3 of 4 (episode 0 first alerts after 299 min). Two of the three "hits" have a 0-minute delay, i.e.
the rolling median was already above threshold before the fault began — coincident false alerts,
not detections. Alerts fire on 2.79 % of fault-free frames of the corrupted stream (bar: 1 %);
on the clean 2016 stream 2.59 % of frames and **145 of 362 days** carry at least one alert.
Occlusion: 36 of 60 episodes alert, typically 10–15 min after onset (the median window).

**Reading.** The per-position detector *ranks* local faults far better than any pooled-embedding
detector (AUROC 0.89 / 0.96 against ≤ 0.74), so the representation is not the bottleneck; the
operating point is. Thresholds calibrated on 2014–15 flag 1.5 % of 2016 frames instead of 1 %
(inter-year drift), and the 30-minute median amplifies rather than suppresses this because
nominal exceedances are temporally clustered. The local score correlates only weakly with
forecast error (Spearman 0.27, against 0.55 for the Isolation Forest score), consistent with a
hardware signal rather than a difficulty signal. Under the pre-registration this is reported as a
negative result; any re-calibration (e.g. drift-aware thresholds) is a new, separately
pre-registered design, not a revision of this one.

## 6. Spatial split of the local detector — pre-registered design (fixed 2026-09-25, before any scoring)

**Motivation (post hoc, stated as such).** Inspecting the §5 outputs after the verdict showed that
91 % of the frames the local detector flags on the clean 2016 stream are blamed on one of the 40
cells that touch the border of the field of view (23 % of the 172 cells), and that those flags
cluster around sunrise and sunset. Mateusz's reading of the top-scoring real frames
(`ch4_admissibility_local_top.pdf`) sorted them into sensing-path events (acquisition faults, a
bird, insects, droplets) and illumination events (sunrise, sunset, sun reflection, heavy rain),
with the illumination events at the rim. The §5 verdict (FAIL on both criteria) stands; this is a
second design, evaluated the same way. No annotated frames exist and none will be made: every
test below is unsupervised, on synthetic faults and on geometry.

**What is unchanged.** Features, projection, per-cell bank, per-cell normalisation, the
calibration frames and days, the 99th-percentile rule and the 30-minute causal median are those
of §5. Re-fitting must reproduce the §5 thresholds (frame 1.7527, alert 1.6356) for the unsplit
score; that reproduction is a check, not a result.

**Split.** *Edge* cells are the valid cells with at least one 4-neighbour outside the field-of-view
mask (40 cells); *interior* cells are the rest (132). Three frame scores: the §5 score (max over all
cells), the interior score (max over interior cells) and the edge score (max over edge cells), each
with the cell attaining it. Each split score gets its own frame threshold, the 99th percentile on
the same 3,000 calibration frames. The alert statistic is the 30-minute causal median of the
*interior* score, thresholded at its 99th percentile on the same 30 calibration days. Per-cell
normalised distances of the 2016 streams are saved (float16) so that any further split can be
computed without re-scoring.

**Pass criteria** — §5's two criteria, applied to the interior detector:

1. *Frame level*: at the interior 1 % threshold, at least 50 % of soiling frames and 50 % of
   occlusion frames flagged, and more than the dark-pixel baseline for each.
2. *Maintenance level*: every one of the 4 soiling episodes raises an interior alert within its
   first 2 hours, and interior alerts fire on at most 1 % of fault-free frames.

**Cost of the split (reported, not a criterion).** Two additional synthetic faults place the same
soiling patches and occlusion disc with centres in the annulus 0.72–0.92 of the field-of-view
radius (`soiling_edge`, `occlusion_edge`). The interior detector is expected to miss them; its
recall on them, against the edge detector's and the dark-pixel baseline's, is the price of
discarding the rim. Faults at the rim are plausible (dirt settling at the dome base, birds on the
housing), so this number bounds what the split gives up.

**Geometric hypothesis (unsupervised, reported, not a criterion).** On the clean 2016 stream, for
every flagged frame the solar elevation and the distance in cells between the blamed cell and the
projected sun position (`cirr/physics/sun.py`, Folsom camera model) are computed. Stated
expectation: among edge flags, at least half have solar elevation below 20° or the blamed cell
within 2 cells of the sun; among interior flags, fewer than half do. This is the proxy for
"illumination event"; it cannot show that interior flags are sensing faults.

**Reporting.** Detection table rows `local, interior cells` and `local, edge cells` beside the §5
row, for the five synthetic faults; maintenance rates, episodes and verdict for the interior
alert; the geometry summary; the top frame per day for each split rendered as in §5.

### 6.1 Result: both criteria FAIL again, with every number improved

Re-fit with `--resume` reproduced the §5 thresholds exactly (1.7527 / 1.6356). Split thresholds:
interior 1.6572 (alert 1.5469), edge 1.7439 (alert 1.6198). Verdict:
`tables/ch4_admissibility_maintenance_verdict_interior.json`.

| | §5 (all cells) | §6 interior | bar |
|---|---|---|---|
| clean frames flagged (probe) | 1.8 % | **0.7 %** | ≈ 1 % |
| soiling flagged | 13.3 % | 21.7 % | ≥ 50 % → FAIL |
| occlusion flagged | 50.5 % | **62.2 %** | ≥ 50 % ✓ |
| AUROC soiling / occlusion | 0.89 / 0.96 | 0.94 / 0.98 | — |
| soiling episodes alerted ≤ 2 h | 3 of 4 | 3 of 4 (episode 0: 291 min) | 4 of 4 → FAIL |
| alerts on fault-free frames | 2.79 % | 1.59 % | ≤ 1 % → FAIL |
| clean days with an alert | 145 of 362 | 97 of 362 | — |

Three of the four soiling "detections" have a 0-minute delay, i.e. the interior alert was already on
before the fault began — coincident alerts, as in §5.

**Cost of the split.** Nothing measurable on the synthetic rim faults: the interior score flags
more of them than the edge score (rim soiling 10.2 % vs 8.4 %, rim occlusion 40.2 % vs 21.7 %),
because the rendered patches extend inward. The edge score is worse than the interior score on
every fault.

**Geometric hypothesis: half confirmed.** Edge flags: 76 % at solar elevation < 20° or within 2
cells of the sun (expected ≥ 50 % ✓; base rate of low sun among all frames 28 %). Interior flags:
69 % (expected < 50 % ✗). The one-cell-deep edge ring is too thin: exploratory depth analysis
(`logs/ch4/admissibility/local_top_core_interim.log`) puts 95 % of clean flags on the outermost
ring and 3.4 % on the second, and the second ring is still dominated by low-sun and wet-dome frames.

**Exploratory, not pre-registered** (inspection of real frames after the verdict, recorded so it
is not mistaken for a test): ranking by the *median* per-cell score instead of the maximum puts
frame-wide sensing faults at the top of the 2016 stream — 7 of the top 12 (one per day) are
acquisition faults with horizontal banding (4) or birds on the dome (3); the banding frame of
23 Dec 07:42 ranks 2nd of 245,538 frames. Ranking by the maximum over cells at least 3 cells from
the rim puts two birds and a person over the camera in its top 3. Figures `local_top_global`,
`local_top_core3`. Testing either statistic requires a new pre-registration (§7) with a synthetic
acquisition fault.
