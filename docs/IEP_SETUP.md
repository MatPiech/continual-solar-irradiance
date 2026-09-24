# Physics-informed (IEP) input for Chapters 4–5

Branch `feature/physics-informed` (feature/adaptation + feature/federated-learning).

## Input

`physics: true` produces the 7-channel best configuration of the IEP paper:
RGB (ImageNet-normalised, circular crop) · sun mask × I_t · Norm B/R cloud · DIS flow (u, v).

| Deviation from the paper code | Why |
|---|---|
| DIS flow t-5 → t from scratch | the paper's warm start was never enabled (`setUseInitialFlow` not set) |
| Sun channel carries I_t in kW/m² | the scale the irradiance history already uses |
| Image t-5 gets the same augmentation replay as t | otherwise augmentation shows up as motion |
| Camera parameters refitted on the centre crop | paper values were for the full 1536 frame |

The flag is stored in the checkpoint's `hparams`. `evaluate.py`,
`related_works/evaluate_lagged_ogd.py` and `fl_benchmark.py` read it (`checkpoint_physics`),
so they build the matching dataset automatically. Other scripts still load 3-channel data.

## Sun-mask camera parameters

`scripts/fit_sun_camera.py --site <site>` detects the sun on clear frames, then fits
azimuth offset, focal length, optical centre, a cubic radial term (`k3`), a mirror flag,
and a clock correction per DST regime. It uses RANSAC to tolerate mis-detected frames.
Overlays and per-month inlier rates are in `results/sun_camera/`.
`scripts/select_sun_mask.py --site <site>` lets you check or adjust the settings by hand
(OpenCV sliders, `s` saves a SITES entry; `--headless` only writes the overlay grid).

| Site | Inliers (<10 px @512) | Median error @128 | Notes |
|---|---|---|---|
| Folsom | 88 % | 0.8 px | paper projection (k3 = 0), no clock correction |
| Poznań | 81.5 % | 0.85 px | image clock +68 min off during standard time |
| Karpicko | 80 % | 0.8 px | image clock −70 min off during DST |

**Finding: the Polish image clocks do not follow the DST switch consistently.** In one regime
per site, image names are about 70 min away from what the sun position says.
Only the difference between regimes can be identified. The irradiance is paired
with each image record, so the forecasting labels are unaffected.
But `*_clear_sky.csv` is keyed by wall-clock time, so the clear-sky value, CSI,
the lit-sample filter, Smart Persistence and the `clear_sky` regime labels are
about 1 h out of phase for that part of the year at both Polish sites. This affects
the existing RGB results too. Not fixed here; it needs a decision.

## Training recipe

`params.yaml`: ResNet50, ImageNet init (SupCon off), BerHu, 128 px, batch 64, lr 3e-4,
Folsom 2014–2015 (146 random days as validation), 2016 held out.
Checkpoints: `checkpoints/resnet50_iep_epoch_XXX.ckpt`; log: `logs/physics/train_iep_folsom.log`.
Run detached (`setsid nohup`) so it outlives the launching shell.

**Bug fixed in the shared recipe:** `A.Affine(translate_percent=(-10, 10))` shifted by up to
10× the image size and blanked about 50 % of augmented training frames. It is now
`(-0.1, 0.1)`, as in the paper. The RGB checkpoint (`resnet50_epoch_015`) was very likely
trained with the bug, so the RGB-vs-IEP comparison mixes representation and augmentation
unless the RGB model is retrained with the same fix:
`physics: false` plus a different `dirpath`, run on another server.

## Experiments

`CKPT=checkpoints/resnet50_iep_epoch_XXX.ckpt bash scripts/bash_scripts/run_physics_eval.sh <1|2|3|4>`
— 6 Ch4 runs (static and conditional × 3 sites, `--no-retention`) and
2 Ch5 runs (FedAvg and FedProto λ=0.1, seed 1234, pure-FL configuration). All other
flags copy the RGB tier-1 runs; results go to `results/physics/`.

Smoke-tested end to end on the epoch-0 checkpoint: static (Poznań, 1 day), conditional
(Karpicko, 1 day) and FedProto (3 sites, 2 days) all finish and write `summary.json`
(logs: `logs/physics/smoke_*.log`). The RGB tier-1 runs took 30–65 min each. The
7-channel input decodes two images per sample, so plan on about 2–2.5× that.
The slices in the launcher are balanced to match.
