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

## 7. Two-channel monitor: exclusion from adaptation and a maintenance alert — pre-registered design (fixed 2026-09-25, before any scoring)

**Motivation (post hoc, stated as such).** Inspection of real 2016 frames after §6 (figures
`local_top_global`, `local_top_core3`) showed that sensing-path events differ in *footprint* and
*duration*, and that each needs a different response (Mateusz's operational framing):

- *short-term* events — acquisition faults (horizontal banding), birds and insects on the dome —
  last one frame to tens of minutes; the frame is useless as training evidence, so the update
  policy should **exclude** it, and nobody needs to be called out;
- *persistent* faults — water droplets on the dome, soiling — last hours to days and do not go away
  by themselves; they call for a **maintenance alert** (clean the dome).

Illumination events (low sun, reflections) sit on the rim and are neither. No annotated frames are
used: every test is on synthetic faults, as in §5–§6.

**Statistics.** From the per-cell normalised distances $z$ of §6 (unchanged features, bank,
normalisation and calibration frames):

- *global* $G$ = median of $z$ over all 172 cells — frame-wide departures (banding, a bird covering
  the dome, a wet dome);
- *core* $C$ = max of $z$ over the 72 cells at chessboard depth ≥ 3 from the field-of-view border —
  compact objects away from the rim (insects, a person over the camera).

**Exclusion flag** $E = [G > t_G] \lor [C > t_C]$, with $t_G$, $t_C$ the 99.5th percentiles of $G$
and $C$ on the same 3,000 calibration frames (nominal union rate ≤ 1 %). A flagged frame is
withheld from the update policy (the `--anomaly-gate` path of Algorithm 1), nothing else.

**Maintenance alert** $M$: causal rolling median of $G$ over the preceding **60 minutes** of the
same day, thresholded at its 99th percentile on the same 30 calibration days. A 60-minute median
cannot be moved by an event shorter than 30 minutes, which is what separates the two channels.

**Synthetic faults** (rendered onto clean 2016 frames like §5's, seeded per episode):

| kind | stands for | rendering | episodes (stream) |
|---|---|---|---|
| `banding` | acquisition fault | 6–20 horizontal bands, 1–6 px high, each with a per-channel gain of 0.6–1.5 and row noise; redrawn every frame | 40 × 1–5 min |
| `occlusion` | bird (existing §5 fault) | dark ellipse, 8–22 % of the image | 60 × 5–30 min |
| `insect` | insect on the dome | dark body 1.5–3.5 px with six legs 3–6 px long, within 0.6 of the radius, drifting 0–1 px per frame | 40 × 2–15 min |
| `wet_dome` | droplets on the dome | 150–350 droplets of radius 1–4 px, uniform over the dome, each a blurred, brightened lens with a dark rim; 80 % of droplets persist within the episode, 20 % redrawn per frame | 20 × 60–240 min |
| `soiling` | soiling (existing §5 fault) | as §5 | 4 × 1–3 days |

The stream schedule uses seed 1 and the same non-overlap rule as §5 (kinds in the order of the
table get precedence: soiling, wet_dome, occlusion, banding, insect). The §5 `droplets` fault is
not part of this stream: it is sparse (20–40 spots) and real wet-dome frames look like `wet_dome`.

**Pass criteria.**

1. *Exclusion* (frame level, detection probe of §5, 3,070 frames per condition): $E$ flags at least
   50 % of `banding` frames and at least 50 % of `occlusion` frames, and at most 2 % of clean 2016
   frames. `insect` is reported without a bar (at 128 × 128 an insect is a few pixels).
2. *Maintenance* (corrupted 2016 stream): $M$ fires within 90 minutes of onset in every `wet_dome`
   episode and within 2 hours in every `soiling` episode, whose statistic must not already be above
   threshold in the 30 minutes before onset (an alert that is already on does not count as a
   detection); $M$ fires on at most 1 % of fault-free frames; and $M$ fires during at most 10 % of
   short-term episodes (`banding`, `occlusion`, `insect`).

**Reported, not criteria.**

- *The exclusion flag in series with the policy*: the damped policy (layer3+4, encoder lr 1e-6) on
  the §7 corrupted stream without and with $E$ as its gate, and on the clean 2016 stream with $E$
  (against the existing ungated clean run); $\Delta$FS with the 95 % day-block bootstrap, MAE on
  fault-free frames, updates made on faulty frames, and frames withheld.
- On the clean 2016 stream: the rate of $E$ and $M$, and the top frames per day for $G$, $C$ and $M$.

**Amendment (2026-09-25, before any scoring).** A visual preview of the rendered faults
(`figures/ch4/ch4_monitor_faults_preview`) showed `banding` far more saturated and `wet_dome`
far more opaque than the real frames that motivated them. Changed before scoring: banding gain
0.8–1.25 (was 0.6–1.5) and row noise σ 0.03 (was 0.04); wet-dome droplet radius 1–3 px (was 1–4),
each droplet blending 60 % refracted and 40 % original sky, dark rim at 25 % (was fully refracted,
40 % rim). No statistic, threshold or criterion changed. `scripts/fit_monitor.py` implements §7.

### 7.1 Result: exclusion flag PASSES, maintenance alert FAILS

Thresholds on 2014–15 calibration: $t_G$ 1.1134, $t_C$ 1.5299 (union flags 0.97 % of calibration
frames), maintenance 1.0112. Verdict: `tables/ch4_admissibility_monitor_verdict.json`.

**Criterion 1 (exclusion): PASS.** Probe of 3,070 frames per condition: banding 85.4 % flagged,
occlusion (bird) 79.2 %, clean 2016 frames 1.2 % (bars ≥ 50 %, ≥ 50 %, ≤ 2 %). Insects 33.0 %
(AUROC 0.94). The two statistics split the work as designed: the global median catches banding
(79.6 %), the core max catches occlusion (78.6 %) and insects (32.4 %).

**Criterion 2 (maintenance): FAIL**, on detection and on false alerts; the separation holds.

- Wet dome: 7 of 19 episodes alerted, 5 of them within 90 min (bar 19 of 19). The wet-dome frames
  are ranked well (probe AUROC 0.92 for E) but their 60-min median of $G$ mostly stays below
  threshold.
- Soiling: 2 of 4 within 2 h (both at 0 min on the first frame of the day, with no alert in the
  preceding 30 min, so they count); one after 54 h, one never.
- Fault-free frames: 3.92 % carry a maintenance alert (bar 1 %); 76 of 362 clean days. The
  2014–15 threshold does not hold on 2016 (clean exclusion rate 1.26 % against 0.97 % on
  calibration; maintenance 3.9 % against 1 %) — the inter-year drift of §5, now in the median.
  Real wet-dome, bird and banding frames in the "fault-free" stream contribute an unknown share.
- Separation: the maintenance alert fires during 3.0 % of short-term episodes (bar ≤ 10 % ✓):
  0.8 % of banding frames, 2.9 % of bird frames, 5.0 % of insect frames.

**Exclusion in series with the damped policy (reported).** Clean stream: E withholds 2,942 frames
(1.2 %); ΔFS −0.05 [−0.10, −0.005], MAE −0.09 %. Corrupted stream: the faults cost the ungated
policy ΔFS −0.06 [−0.16, +0.02] (not significant); E cuts the updates made on faulty frames by a
third (4,423 → 2,938) and changes fault-free-frame skill by ΔFS −0.05 [−0.10, −0.006] with MAE
−0.18 % [−0.28, −0.10]. Against the §4 Isolation Forest gate (ΔFS −0.15 to −0.16), E costs a third
as much skill and improves MAE; neither effect is large, because faults with valid labels do not
damage the policy in the first place.

**Amendment (2026-09-25, after the §7 verdict — affects future runs only).** On Mateusz's
instruction the synthetic fault set was changed: `droplets` (§5) and `wet_dome` (§7) are retired
(`RETIRED_KINDS`; still rendered by `corrupt()` so every scored stream reproduces), and the
occluder now draws a colour (black, brown, light grey, dark green) and a shape (ellipse, rotated
ellipse, 3–4-lobe blob, thin bar) per episode instead of a black ellipse. The §5–§7 verdicts stand
as scored with the old faults. With `wet_dome` retired, `PERSISTENT` is soiling only; any re-scoring
under the new fault set is a new pre-registration, not a revision of §7. Figures
`ch4_admissibility_fault_mosaic` and `ch4_admissibility_faults` show the new set.

## 8. Fault dose and duration — pre-registered design (fixed 2026-09-25, before any scoring)

**Question.** Does input admissibility matter for the update policy, and in which direction? The
§4 and §7 gate-in-series runs found small effects at one dose (3–4 % of frames faulty), which
leaves the objection that the faults were too rare. This study varies the dose and the duration
of synthetic faults, with the damped policy (layer3+4, encoder lr 1e-6), the static model and the
§7 exclusion flag $E$ (thresholds unchanged, `results/adaptation/gate/monitor/monitor.json`).

**Streams** (Folsom 2016 daylight population; faults rendered by `corrupt()` with the amended
fault set: soiling, occlusion with varied colour/shape, banding, insect; seeds fixed):

| stream | faults | construction |
|---|---|---|
| `r01`, `r05`, `r15` | 1, 5, 15 % of daylight frames faulty | half of the faulty frames from whole-day soiling episodes of 1–3 days, half from short-term episodes in the episode ratio occlusion : banding : insect = 3 : 2 : 2 with §7 durations; episodes added until the budget is met; seed = rate |
| `cont_soiling` | one soiling episode (one patch set) on every daylight frame from 2016-07-01 to 2016-08-15 inclusive, then "cleaned" | continuous fault with an end |
| `cont_occluder` | one occluder (one shape, colour and interior position) on every daylight frame from 2016-09-01 to year end | permanent fault |

**Arms per stream** (single seed): *static* (policy with $T_1 = T_2 = \infty$, no updates),
*damped ungated*, *damped gated* ($E$ withholds flagged frames from updates, Algorithm 1 line 1).
Clean references are the existing runs (static baseline, damped, damped with $E$).

**Reported per stream.** MAE and forecast skill on fault-free and on faulty frames for each arm;
day-block bootstrap contrasts (2,000 resamples, 95 % for ΔFS, 90 % for relative MAE): (a) ungated
vs static on faulty frames, (b) gated vs ungated on faulty and on fault-free frames, (c) ungated
vs the clean-stream damped run on fault-free frames; frames withheld, updates, updates on faulty
frames. For the continuous streams additionally: weekly series of faulty-frame MAE per arm, of
the withheld share and of the maintenance-alert share; faulty-frame MAE in the first vs the last
two weeks of the fault; alert share in the 14 days before, during, and the 14 days after.

**Predictions** (each is confirmed or refuted; none is a pass criterion for a detector):

- P1 *Contamination stays small*: ungated vs clean on fault-free frames within ±0.10 FS points at
  every dose.
- P2 *Adaptation learns around persistent faults*: on faulty frames the ungated policy beats the
  static model at 5 % and 15 %, by more at 15 % than at 5 %; the gated policy lies between static
  and ungated.
- P3 *Exclusion starves adaptation under a continuous fault*: on `cont_soiling` $E$ withholds
  more than 30 % of faulty frames; the ungated policy's faulty-frame MAE is lower in the last two
  weeks than in the first two; the gated policy's is not (by less than half the ungated drop).
- P4 *The maintenance alert follows a continuous fault and clears*: alert share > 50 % of frames
  during `cont_soiling`, < 5 % in the 14 days before, and back below 5 % within the 14 days after
  cleaning.
- P5 *A permanent occluder is the failure case for exclusion*: on `cont_occluder` $E$ withholds
  more than 50 % of frames from onset to year end and the alert share exceeds 50 %; the ungated
  policy's faulty-frame MAE converges toward the clean-stream level (last two weeks within 5 % of
  the clean damped run on the same dates), the gated policy's does not.

**Cost.** Per stream: gate scoring ≈ 12 min GPU; three policy runs ≈ 1.2 h in parallel. Five
streams across four servers ≈ one evening. Code: `scripts/make_dose_schedules.py`,
`scripts/fit_monitor.py --schedule … --thresholds …`, `scripts/admissibility_study.py dose`.
