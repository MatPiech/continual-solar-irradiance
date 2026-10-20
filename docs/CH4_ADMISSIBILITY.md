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
