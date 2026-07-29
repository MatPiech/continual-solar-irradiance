# Continual Solar Irradiance Forecasting

Online and federated adaptation for sky-image solar irradiance forecasting. The `cirr` library implements forecasting models, physics-guided gating, test-time adaptation (TTA), and federated learning (FL) for deployment at sites where ground truth arrives with a causal delay.

---

## Project objective

Forecast short-horizon global horizontal irradiance (GHI) from sky images and recent irradiance history at the edge:

| Concern | What the codebase does |
|---------|-------------------------|
| **Delayed feedback** | Predict at *t*; ground truth usable after `horizon_steps` (default 15 samples) |
| **Online adaptation** | OGD, sliding window, SM-RLS, CSI gating, TTA — update only when triggers fire |
| **Federated learning** | Multi-site aggregation (FedAvg, FedBN, FedPer, …) on a sync interval decoupled from daily noise |

Benchmark sites: **Folsom** (US source), **Karpicko**, **Poznan**.

---

## Architecture (library vs scripts)

```
scripts/*.py          CLI: argparse/click, paths, result files
        │
        ▼
cirr/evaluation/      load_eval_model, load_eval_days, run_delayed_online_eval
cirr/training/        ogd, sm_rls, day_loop, replay
cirr/continual/       gating, horizon FIFO buffer
cirr/fl/              server aggregation, prototypes
cirr/tta/             MEMO, NOTE, dual-phase
cirr/models/          Forecaster, checkpoint loading
cirr/data/            datasets, transforms, load_dataset()
cirr/physics/         clear-sky GHI, CSI, thresholds
```

**Design rule:** adaptation behavior lives in `cirr/`; scripts wire checkpoints, datasets, and `--save-results`. Online eval scripts share `run_delayed_online_eval()` and `cirr/evaluation/runner.py` bootstrap helpers.

---

## Repository layout

```
continual-solar-irradiance/
├── cirr/                 # Installable package
├── scripts/              # Experiment CLIs (see table below)
├── tests/                # Smoke tests (ogd, sm_rls, gating)
├── params.yaml           # DVC pipeline parameters
├── CONTEXT.md            # Domain glossary + module index
└── WACV.md               # Research checklist
```

### Scripts inventory

| Script | Purpose |
|--------|---------|
| **Data & training** | |
| `scripts/dvc/get_data.sh` | Download Folsom raw data (Zenodo) |
| `scripts/dvc/convert_timestamps.py` | Rename images to UTC timestamps |
| `scripts/dvc/clean_dataframe.py` | Merge irradiance + image paths |
| `scripts/dvc/export_periods.py` | Build `train_periods.pickle` |
| `scripts/dvc/train_forecaster.py` | Train Forecaster (Lightning) |
| `scripts/pretrain_supcon.py` | SupCon encoder pretraining |
| `scripts/clear_sky.py` | Generate site `*_clear_sky.csv` files |
| **Evaluation & adaptation** | |
| `scripts/evaluate.py` | Baseline + MEMO/NOTE TTA |
| `scripts/evaluate_dual_phase.py` | Dual-phase OGD + TTA |
| `scripts/evaluate_paper_lagged_ogd.py` | Paper Algorithm 1 OGD |
| `scripts/evaluate_sliding_window.py` | Sliding-window OGD |
| `scripts/evaluate_sm_rls.py` | Sherman–Morrison RLS head |
| `scripts/evaluate_gated.py` | CSI prototype gating |
| `scripts/local_training.py` | Per-day local continual training |
| `scripts/local_gated_training.py` | Per-day local continual training with delayed prototype gating |
| **Federated & benchmarks** | |
| `scripts/fl_benchmark.py` | Multi-site FL benchmark |
| `scripts/fl_edge_overhead_benchmark.py` | FL round platform cost (edge devices) |
| `scripts/edge_adaptation_benchmark.py` | Adaptation latency / throughput |
| `scripts/run_baselines.py` | Orchestrate static / OGD / gated / FL runs |
| `scripts/smart_persistence_model.py` | Smart-persistence baseline |
| **Batch wrappers** (`scripts/bash_scripts/`) | |
| `run_fl_benchmark.sh` | Sweep FL algorithms |
| `generate_clear_sky_all.sh` | Clear-sky CSVs for all sites |
| `calculate_smart_persistence_all.sh` | Smart persistence for all sites |

### Key APIs

| Module | Functions |
|--------|-----------|
| `cirr/data/loading.py` | `load_dataset(path, split="test"\|"training")` |
| `cirr/evaluation/runner.py` | `load_eval_model`, `load_eval_days`, `print_eval_plan`, `save_online_eval_results` |
| `cirr/evaluation/online.py` | `run_delayed_online_eval`, `infer_one` |
| `cirr/continual/buffer.py` | `reveal_after_horizon`, `DelayedFeedbackBuffer` |
| `cirr/training/ogd.py` | Algorithm 1 OGD, sliding-window steps |
| `cirr/continual/gating.py` | CSI prototype gating + `eval_gated_online` |

---

## Datasets

### Prepared directory

Each site (e.g. `data/poznan_prepared/`) contains:

| File / dir | Role |
|------------|------|
| `images/` | Sky images referenced by period records |
| `periods.pickle` | Test split index |
| `train_periods.pickle` | Training split index |
| `cleaned_irradiance.csv` | Preprocessed irradiance series |

Forecasting window: **15 min horizon**, **5 min history step**, **4 history samples** (`params.yaml`).

### Sample tuple

```python
image, history, ghi_clear, target, timestamp = dataset[i]
```

| Field | Shape | Meaning |
|-------|-------|---------|
| `image` | `[C,H,W]` | RGB (3ch) or RGB+flow (5ch) |
| `history` | `[4]` | Normalised past irradiance |
| `ghi_clear` | scalar | Clear-sky GHI |
| `target` | `[1]` | Normalised ground truth |
| `timestamp` | str | `YYYYMMDD_HHMMSS` |

```python
from pathlib import Path
from cirr.data.loading import load_dataset

ds = load_dataset(Path("data/poznan_prepared"), split="test", optical_flow="none")
```

### Folsom download

```bash
dvc repro get_data convert_timestamps clean_dataframe export_periods
cp data/folsom_prepared/train_periods.pickle data/folsom_prepared/periods.pickle
```

Or download raw data only: `bash scripts/dvc/get_data.sh`.

Other sites use the same prepared layout after site-specific preprocessing. See [Start from scratch](#start-from-scratch) for the full Folsom pipeline.

---

## Installation

**Prerequisites:** Python 3.10–3.12, CUDA optional (recommended for training), ~50 GB disk for Folsom raw images.

```bash
git clone https://github.com/MatPiech/continual-solar-irradiance.git
cd continual-solar-irradiance
pip install -e ".[dev]"
pytest tests/ -q
```

---

## Start from scratch

End-to-end path from raw data to WACV-style evaluation. Steps map to [WACV.md](WACV.md) phases.

| Step | WACV phase | Command / artifact |
|------|------------|-------------------|
| 1 Prepare data | Phase 1 | `dvc repro …` → `train_periods.pickle` |
| 2 Clear-sky CSVs | Phase 1.3 / 4 | `generate_clear_sky_all.sh` |
| 3 SupCon (optional) | Phase 3.1 | `pretrain_supcon.py` |
| 4 Train Forecaster | Phase 2 | `train_forecaster.py` → `checkpoints/` |
| 5 Static baseline | Phase 6.1 | `evaluate.py` |
| 6 Online adaptation | Phases 4–5 | `evaluate_gated.py`, `evaluate_paper_lagged_ogd.py`, … |
| 7 Federated | Phase 5 | `fl_benchmark.py` |
| 8 Baseline suite | Phase 6.1 | `run_baselines.py` |

```
Install → Prepare data → Clear-sky CSVs → (optional SupCon) → Train Forecaster → Evaluate
```

### Step 1 — Prepare Folsom (source site)

The DVC pipeline downloads Folsom from [Zenodo 2826939](https://zenodo.org/records/2826939), renames images, merges irradiance, and exports period indices.

```bash
dvc repro get_data convert_timestamps clean_dataframe export_periods
```

This writes under `data/folsom_prepared/`:

| Output | Role |
|--------|------|
| `images/` | Timestamped sky images |
| `cleaned_irradiance.csv` | Aligned irradiance + image paths |
| `train_periods.pickle` | Period index for training |

Eval scripts load `periods.pickle` for the test split. After export, link the same index file:

```bash
cp data/folsom_prepared/train_periods.pickle data/folsom_prepared/periods.pickle
```

For **Folsom**, `ForecastingDataModule` splits years internally: 2014–2015 train/val, **2016 test** (matches WACV US Years 1–2 pretrain / Year 3 eval).

**Other sites** (Poznan, Karpicko, …) must follow the same prepared layout (`images/`, `periods.pickle`, `train_periods.pickle`, `cleaned_irradiance.csv`) using your site-specific preprocessing. The DVC stages above are Folsom-only.

Period geometry (fixed in `params.yaml`): **15 min** forecast horizon, **5 min** history step, **4** history samples.

### Step 2 — Clear-sky reference (required for gating)

Gated adaptation and CSI routing need `*_clear_sky.csv` per site (`cirr/physics/clear_sky.py` loads `*clear_sky.csv` from the dataset directory).

```bash
bash scripts/bash_scripts/generate_clear_sky_all.sh
```

Or generate one site:

```bash
python scripts/clear_sky.py \
  --latitude 52.42 --longitude 16.96 --timezone "Europe/Warsaw" \
  --start "2024-01-01 00:00:00" --end "2024-12-31 23:59:00" \
  --output-path data/poznan_prepared/poznan_clear_sky.csv
```

### Step 3 — Optional: SupCon encoder pretraining (WACV Phase 3)

Physically-guided contrastive pretraining structures the vision encoder by CSI bin (`cirr/losses/supcon.py`). Trains on Folsom **training** split (years 1–2 after internal split):

```bash
python scripts/pretrain_supcon.py \
  --data-root data/folsom_prepared \
  --epochs 5 \
  --batch-size 64 \
  --output checkpoints/supcon_encoder.pt
```

Output: `checkpoints/supcon_encoder.pt`. Use it in forecaster training:

```yaml
# params.yaml
train_forecaster:
  model:
    encoder_checkpoint: checkpoints/supcon_encoder.pt
```

When set, ImageNet init is skipped and the SupCon encoder is loaded before `train_forecaster.py` runs.

### Step 4 — Train the Forecaster (WACV Phase 2)

Hyperparameters live in `params.yaml` under `train_forecaster` (model, trainer, datamodule, callbacks). Training reads them via DVC params:

```bash
python scripts/dvc/train_forecaster.py --data-root data/folsom_prepared
```

Checkpoints go to `checkpoints/epoch_XXX.ckpt` (best validation loss). Tune via `params.yaml`:

| Knob | Default | Notes |
|------|---------|-------|
| `model.model_name` | `timm-resnet50` | Vision backbone |
| `model.loss_function` | `Berhu` | Regression loss |
| `datamodule.optical_flow` | `none` | Set `farneback` for 5-channel RGB+flow |
| `trainer.max_epochs` | `30` | Early stopping monitors `val_loss` |
| `mlflow.enabled` | `false` | Set `true` to log metrics to local `mlruns/` |

Enable MLflow in `params.yaml`:

```yaml
train_forecaster:
  mlflow:
    enabled: true
```

View runs:

```bash
mlflow ui --backend-store-uri mlruns
```

Then open http://127.0.0.1:5000. Optional knobs under `train_forecaster.mlflow`: `tracking_uri`, `experiment_name`, `run_name`.

### Step 5 — Static baseline (WACV Phase 6.1)

Verify the checkpoint before any adaptation:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/epoch_007.ckpt \
  --dataset-path data/folsom_prepared \
  --num-days 30 \
  --save-results
```

Smart-persistence lower bound (no model):

```bash
python scripts/smart_persistence_model.py \
  --periods-path data/folsom_prepared/periods.pickle \
  --clear-sky-path data/folsom_prepared/folsom_clear_sky.csv \
  --num-days 30 \
  --save-results
```

### Step 6 — Online adaptation eval (delayed ground truth)

All scripts below assume **`horizon_steps=15`** (15-minute label delay) unless noted. They use `run_delayed_online_eval()` from `cirr/evaluation/online.py`.

**Ungated OGD** (continuous adaptation baseline):

```bash
python scripts/evaluate_paper_lagged_ogd.py \
  --checkpoint checkpoints/epoch_007.ckpt \
  --dataset-path data/folsom_prepared \
  --horizon-steps 15 --save-results
```

**Gated adaptation** (WACV Phase 4 — CSI prototypes + dual threshold):

```bash
python scripts/evaluate_gated.py \
  --checkpoint checkpoints/epoch_007.ckpt \
  --dataset-path data/folsom_prepared \
  --fit-thresholds \
  --horizon-steps 15 \
  --save-results
```

Other methods: `evaluate_sliding_window.py`, `evaluate_sm_rls.py`, `evaluate_dual_phase.py`, TTA via `evaluate.py --method memo|note`.

### Step 7 — Federated benchmark (WACV Phase 5)

Multi-site FL with decoupled sync rounds (`scripts/fl_benchmark.py`):

```bash
python scripts/fl_benchmark.py \
  --checkpoint checkpoints/epoch_007.ckpt \
  --dataset data/folsom_prepared \
  --dataset data/karpicko_prepared \
  --dataset data/poznan_prepared \
  --algorithm fedavg \
  --sync-interval 7 \
  --num-days 90 \
  --save-results results/fedavg_7day
```

Algorithms: `static`, `isolated`, `fedavg`, `fedbn`, `fedper`, `fedprox`, `ditto`, `fedema`, `fedproto`, `gated`.

### Step 8 — Run the WACV baseline suite

Orchestrates static, ungated OGD, gated, and optional FL runs ([WACV.md Phase 6.1](WACV.md)):

```bash
python scripts/run_baselines.py \
  --checkpoint checkpoints/epoch_007.ckpt \
  --dataset-path data/poznan_prepared \
  --baseline static \
  --baseline ungated_ogd \
  --baseline gated
```

Add `--dry-run` to print commands without executing.

### Smoke test (no data download)

With any `.ckpt` and a prepared dataset path:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/epoch_007.ckpt \
  --dataset-path data/poznan_prepared \
  --num-days 3
```

---

## Usage reference

Shared eval flags: `--device cuda`, `--start-day 0101`, `--num-days 30`, `--save-results`. Run `--help` on any script for full options.

### Static eval and TTA

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared

python scripts/evaluate.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared \
  --method memo --n-augmentations 4 --num-days 3
```

Dual-phase OGD+TTA: `scripts/evaluate_dual_phase.py`.

### Online adaptation (delayed GT)

These use `run_delayed_online_eval()` internally:

```bash
python scripts/evaluate_paper_lagged_ogd.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared \
  --horizon-steps 15 --t1 0.01 --t2 0.1

python scripts/evaluate_sliding_window.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared

python scripts/evaluate_sm_rls.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared

python scripts/evaluate_gated.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared \
  --fit-thresholds
```

### Local and federated training

```bash
python scripts/local_training.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared

python scripts/local_gated_training.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared \
  --gate-calibration-samples 300 \
  --gate-horizon-steps 3

python scripts/fl_benchmark.py \
  --checkpoint checkpoints/epoch_007.ckpt \
  --dataset data/folsom_prepared \
  --dataset data/karpicko_prepared \
  --dataset data/poznan_prepared \
  --algorithm fedavg \
  --sync-interval 7 \
  --num-days 90 \
  --save-results results/fedavg_7day
```

FL algorithms: `static`, `isolated`, `fedavg`, `fedbn`, `fedper`, `fedprox`, `ditto`, `fedema`, `fedproto`, `gated`.

Run multiple baselines:

```bash
python scripts/run_baselines.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared \
  --baseline static --baseline ungated_ogd --baseline gated
```

### Edge latency benchmark

```bash
python scripts/edge_adaptation_benchmark.py \
  --checkpoint checkpoints/epoch_024.ckpt \
  --dataset-path data/poznan_prepared \
  --methods baseline,memo,note,paper_ogd,sliding,sm_rls \
  --num-samples 512
```

Use `--input-source random` to profile compute without dataset I/O.

FL platform overhead (payload size, aggregation latency):

```bash
python scripts/fl_edge_overhead_benchmark.py \
  --checkpoint checkpoints/epoch_007.ckpt \
  --device-profile raspberry-pi
```

---

## Links

- Repository: https://github.com/MatPiech/continual-solar-irradiance
- Package: `cirr` (`pyproject.toml`)
