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
bash scripts/dvc/get_data.sh   # Zenodo 2826939
dvc repro                      # clean → periods → train (needs dvc + dev deps)
```

Other sites use the same prepared layout after site-specific preprocessing.

---

## Installation

```bash
git clone https://github.com/MatPiech/continual-solar-irradiance.git
cd continual-solar-irradiance
pip install -e ".[dev]"
pytest tests/ -q
```

---

## Usage

All eval scripts need `--checkpoint` and `--dataset-path`. Shared flags: `--device cuda`, `--start-day 0101`, `--num-days 30`, `--save-results`.

Run `--help` on any script for full options.

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
