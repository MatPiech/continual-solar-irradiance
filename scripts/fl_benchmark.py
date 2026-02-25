"""FL Benchmark: Federated Learning as a Spatial Regularizer.

Methodology
-----------
This benchmark evaluates Federated Learning algorithms on two orthogonal axes:

  Source Retention  - Can FL prevent the source-site model from overfitting to
                      daily stochastic noise and *degrading* over time?
  Target Plasticity - Can FL accelerate a model's adaptation to a new geographic
                      site that it has never seen before?

Client Setup
------------
  All clients are treated identically: same adaptation config, same training
  loop.  The first dataset passed is labelled "source" in the output for
  evaluation purposes only.  Optionally, the source client can be given
  different adaptation hyperparameters (--source-adapt-lr / --source-adapt-epochs)
  to model a scenario where the source site has a different training regime.

FL Round Schedule (Decoupled from Daily Updates)
------------------------------------------------
Clients train locally for `--sync-interval` days (default: 7), then a single
FL aggregation round fires.  This strips out accumulated daily noise before it
compounds into catastrophic forgetting.

Algorithms
----------
  isolated  No federation — lower bound, proves source degrades and targets lag.
  fedavg    Standard FedAvg: average all weights every sync interval.
  fedbn     Aggregate all layers *except* BatchNorm statistics, preserving
            per-site covariate normalization.
  fedper    Aggregate encoder only; each client's network_head stays local
            (personalised regression mapping).

Usage
-----
  python scripts/fl_benchmark.py \\
      --checkpoint checkpoints/epoch_007.ckpt \\
      --source-dataset data/folsom_prepared \\
      --target-dataset data/karpicko_prepared \\
      --target-dataset data/poznan_prepared \\
      --algorithm fedbn \\
      --sync-interval 7 \\
      --num-days 90 \\
      --save-results results/fedbn_7day
"""

import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import click
import numpy as np
import pandas as pd
import torch

from cirr.datamodules.forecasting import ForecastingDataModule
from cirr.fl.server import AGGREGATION_FNS, _is_bn_key, _is_head_key
from cirr.models.forecaster import Forecaster
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_test_dataset(dataset_path: Path) -> Any:
    """Load the test split of a prepared dataset."""
    dm = ForecastingDataModule(
        root_data_path=dataset_path,
        periods_path=dataset_path / "periods.pickle",
        augment=False,
        image_size=[128, 128],
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        batch_size=1,
        workers=0,
        seed=42,
    )
    dm.setup(None)
    return dm._test_dataset


def _load_checkpoint(checkpoint_path: Path, device: str) -> Forecaster:
    """Instantiate a Forecaster and load weights from *checkpoint_path*."""
    data = torch.load(checkpoint_path, map_location=device)
    hparams = data.get("hyper_parameters", {})
    model = Forecaster(
        model_name=hparams.get("model_name", "timm-resnet50"),
        pretrained=False,
        input_channels=hparams.get("input_channels", 3),
        loss_function=hparams.get("loss_function", "Berhu"),
        lr=hparams.get("lr", 1e-3),
        lr_patience=hparams.get("lr_patience", 5),
    ).to(device)
    model.load_state_dict(data["state_dict"])
    return model


# ---------------------------------------------------------------------------
# Per-day inference / training
# ---------------------------------------------------------------------------


def _predict_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference for every sample in *day_indices*.

    Returns
    -------
    predictions : np.ndarray, shape (N,)
    targets     : np.ndarray, shape (N,)
    """
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for idx in day_indices:
            images, irr_hist, target, _ = dataset[idx]
            images = images.unsqueeze(0).to(device)
            irr_hist = irr_hist.unsqueeze(0).to(device)
            mean, _ = model(images, irr_hist)
            preds.append(mean.cpu().item())
            tgts.append(target.cpu().item())
    return np.array(preds), np.array(tgts)


def _train_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    batch_size: int,
    epochs: int,
    device: str,
) -> float:
    """Train *model* for *epochs* epochs on *day_indices*.

    Returns
    -------
    avg_loss : float
    """
    if not day_indices:
        return 0.0

    samples = [(dataset, idx) for idx in day_indices]
    random.shuffle(samples)
    batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]

    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_loss = 0.0
    count = 0

    for _ in range(epochs):
        for batch in batches:
            imgs = torch.stack([dataset[idx][0] for _, idx in batch]).to(device)
            hist = torch.stack([dataset[idx][1] for _, idx in batch]).to(device)
            tgts = torch.stack([dataset[idx][2] for _, idx in batch]).to(device)

            optimizer.zero_grad()
            mean, var = model(imgs, hist)
            if var is not None:
                loss = model.loss_fn(mean, tgts, var)
            elif model.loss_name == "Berhu":
                loss = model.loss_fn(mean * 1000, tgts * 1000)
            else:
                loss = model.loss_fn(mean, tgts)

            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            count += 1

    return total_loss / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-client BN / head state tracking for FedBN and FedPer
# ---------------------------------------------------------------------------


def _extract_local_keys(state_dict: dict[str, Any], key_filter) -> dict[str, Any]:
    """Return a sub-dict of *state_dict* for keys matching *key_filter*."""
    return {k: v.clone() for k, v in state_dict.items() if key_filter(k)}


def _restore_local_keys(
    aggregated: dict[str, Any],
    local_state: dict[str, Any],
) -> dict[str, Any]:
    """Overwrite *aggregated* with *local_state* entries (in-place copy)."""
    merged = copy.deepcopy(aggregated)
    for k, v in local_state.items():
        merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Warmup cache helpers
# ---------------------------------------------------------------------------


def _compute_warmup_cache_key(
    checkpoint_path: Path,
    clients: list[tuple[str, Path]],
    warmup_days: int,
    start_day: str,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_batch_size: int,
    adapt_epochs: int,
    seed: int,
) -> str:
    """Compute a short hex digest that uniquely identifies a warmup configuration.

    *algorithm* is intentionally excluded: the warmup phase is algorithm-agnostic
    (all methods do identical isolated local training), so the resulting per-client
    weights are the same regardless of which FL algorithm follows.
    """
    config = {
        "checkpoint": str(checkpoint_path),
        "clients": [(name, str(path)) for name, path in clients],
        "warmup_days": warmup_days,
        "start_day": start_day,
        "adapt_lr": adapt_lr,
        "adapt_weight_decay": adapt_weight_decay,
        "adapt_max_grad_norm": adapt_max_grad_norm,
        "adapt_batch_size": adapt_batch_size,
        "adapt_epochs": adapt_epochs,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_fl_benchmark(
    checkpoint_path: Path,
    clients: list[tuple[str, Path]],
    algorithm: str,
    sync_interval: int,
    num_days: int,
    start_day: str,
    warmup_days: int,
    warmup_cache_dir: Path | None,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_batch_size: int,
    adapt_epochs: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Run the N-client FL benchmark.

    All clients use identical adaptation config.

    Parameters
    ----------
    checkpoint_path     : Path to initial pretrained checkpoint (all clients start here).
    clients             : Ordered list of (name, dataset_path) tuples.
    algorithm           : One of "isolated", "fedavg", "fedbn", "fedper".
    sync_interval       : Days between FL aggregation rounds (e.g. 7 or 14).
    num_days            : Total days to simulate (-1 = all).
    start_day           : MMDD string (e.g. "0101").
    warmup_days         : Number of initial days where all methods act as isolated
                          (no federation).  At the end of the warmup period a single
                          forced aggregation round fires before resuming the normal
                          sync_interval schedule.  0 = no warmup.
    warmup_cache_dir    : Optional directory for caching per-client model states at
                          end of warmup.  The cache key is derived from all warmup-
                          relevant hyperparameters (checkpoint, datasets, warmup_days,
                          lr, …) but NOT the algorithm, so the same snapshot is reused
                          across different algorithm runs with identical warmup configs.
                          Pass None to disable caching.
    adapt_lr            : Local learning rate (all clients).
    adapt_weight_decay  : AdamW weight decay.
    adapt_max_grad_norm : Gradient clipping (-1 = disabled).
    adapt_batch_size    : Mini-batch size.
    adapt_epochs        : Local epochs per day (all clients).
    device              : "cuda" or "cpu".
    seed                : RNG seed.

    Returns
    -------
    results : dict with keys:
        "algorithm", "sync_interval", "client_names",
        "per_day"         - list[dict] with per-day per-client metrics,
        "per_round"       - list[dict] with per-FL-round aggregation stats,
        "overall_metrics" - dict[client_name -> metric_dict].
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(clients) < 2:
        raise ValueError("At least two clients are required.")

    algorithm = algorithm.lower()
    if algorithm not in ("isolated", *AGGREGATION_FNS.keys()):
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: isolated, {', '.join(AGGREGATION_FNS)}")

    client_names = [name for name, _ in clients]

    print(f"\n{'='*70}")
    print(f"  FL Benchmark — algorithm: {algorithm.upper()}  sync_interval: {sync_interval} days  warmup: {warmup_days} days")
    print(f"{'='*70}")
    for name, path in clients:
        print(f"  {name:20s}  lr={adapt_lr}  epochs={adapt_epochs}  path={path}")
    print()

    # --- Load datasets ---
    print("Loading datasets...")
    datasets = []
    day_maps = []
    for name, path in clients:
        ds = _load_test_dataset(path)
        dm = group_indices_by_day(ds._periods)
        datasets.append(ds)
        day_maps.append(dm)
        print(f"  {name}: {len(ds._periods)} periods")

    # --- Load models (all clients start from the same checkpoint) ---
    print(f"\nLoading checkpoint: {checkpoint_path}")
    models = [_load_checkpoint(checkpoint_path, device) for _ in clients]
    print(f"  All {len(clients)} clients initialised from the same checkpoint.\n")

    # --- Day schedule ---
    all_days = generate_year_days()
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day '{start_day}'. Expected MMDD format.")
    all_days = all_days[all_days.index(start_day) :]
    if num_days > 0:
        all_days = all_days[:num_days]
    print(f"Simulating {len(all_days)} days starting {all_days[0]} → {all_days[-1]}\n")

    # --- Progress table header ---
    _col_w = 8 + 2 + 8 + 2 + 5  # MAE(8) + 2sp + RMSE(8) + 2sp + N(5) = 25
    _col_w = max(_col_w, max(len(n) for n in client_names))
    _header_names = "  ".join(f"{n:^{_col_w}s}" for n in client_names)
    _header_metrics = "  ".join(f"{'MAE':>8}  {'RMSE':>8}  {'N':>5}" for _ in client_names)
    _sep = "  ".join(["-" * _col_w] * len(client_names))
    print(f"  {'Day':<6}  {_header_names}")
    print(f"  {'':6}  {_header_metrics}")
    print(f"  {'------'}  {_sep}")

    # --- Per-client state for FedBN / FedPer ---
    local_bn: list[dict[str, Any]] = [{} for _ in clients]
    local_head: list[dict[str, Any]] = [{} for _ in clients]

    per_day: list[dict[str, Any]] = []
    per_round: list[dict[str, Any]] = []
    days_since_sync = 0
    fl_round_idx = 0
    last_global_state_dict: dict[str, Any] | None = None  # updated after every FL round
    in_warmup = warmup_days > 0 and algorithm != "isolated"

    # Accumulators for final overall metrics (avoid a second inference pass)
    _all_preds: list[list[np.ndarray]] = [[] for _ in clients]
    _all_tgts: list[list[np.ndarray]] = [[] for _ in clients]

    # -----------------------------------------------------------------------
    # Warmup cache — load if available, skipping the warmup simulation
    # -----------------------------------------------------------------------
    _warmup_cache_loaded = False
    _warmup_cache_key: str | None = None
    _warmup_cache_file: Path | None = None

    _use_cache = warmup_days > 0 and algorithm != "isolated" and warmup_cache_dir is not None
    if _use_cache:
        _warmup_cache_key = _compute_warmup_cache_key(
            checkpoint_path,
            clients,
            warmup_days,
            start_day,
            adapt_lr,
            adapt_weight_decay,
            adapt_max_grad_norm,
            adapt_batch_size,
            adapt_epochs,
            seed,
        )
        _warmup_cache_file = warmup_cache_dir / f"{_warmup_cache_key}.pt"

        if _warmup_cache_file.exists():
            print(f"  Warmup cache HIT  ({_warmup_cache_file})")
            print(f"  Skipping {warmup_days} days of isolated warmup training.\n")
            _cache = torch.load(_warmup_cache_file, map_location=device, weights_only=False)
            for i, model in enumerate(models):
                model.load_state_dict(_cache["client_states"][i])
            per_day = _cache["per_day"]
            _all_preds = _cache["all_preds"]
            _all_tgts = _cache["all_tgts"]
            # Keep in_warmup=True so that warmup_ended_today fires on the
            # first iteration (day_num == warmup_days), triggering the forced
            # aggregation round naturally — identical to the non-cached path.
            _warmup_cache_loaded = True
        else:
            print(f"  Warmup cache MISS (key={_warmup_cache_key})")
            print(f"  Will save snapshot to {_warmup_cache_file} after warmup ends.\n")

    # When cache was loaded, skip warmup days; day_num starts at warmup_days
    # so the existing warmup_ended_today logic fires on the first iteration.
    _loop_start = warmup_days if _warmup_cache_loaded else 0

    # -----------------------------------------------------------------------
    # Main loop — identical treatment for every client
    # -----------------------------------------------------------------------
    for day_num, day_key in enumerate(all_days[_loop_start:], start=_loop_start):
        # Check whether warmup just ended (transition on this day)
        warmup_ended_today = in_warmup and day_num >= warmup_days
        if warmup_ended_today:
            in_warmup = False

        day_record: dict[str, Any] = {"day": day_key, "day_num": day_num, "fl_round_after": None, "in_warmup": in_warmup}
        losses: list[float] = []

        for i, (name, model) in enumerate(zip(client_names, models, strict=False)):
            indices = day_maps[i].get(day_key, [])
            day_record[f"{name}_n"] = len(indices)

            # Step 1: Predict
            preds, tgts = _predict_day(model, datasets[i], indices, device)
            metrics = compute_metrics(preds, tgts) if len(preds) > 0 else {}
            day_record[f"{name}_mae"] = metrics.get("mae", float("nan"))
            day_record[f"{name}_rmse"] = metrics.get("rmse", float("nan"))
            if len(preds) > 0:
                _all_preds[i].append(preds)
                _all_tgts[i].append(tgts)

            # Step 2: Train — same loop for all clients
            loss = _train_day(
                model,
                datasets[i],
                indices,
                lr=adapt_lr,
                weight_decay=adapt_weight_decay,
                max_grad_norm=adapt_max_grad_norm,
                batch_size=adapt_batch_size,
                epochs=adapt_epochs,
                device=device,
            )
            losses.append(loss)

        per_day.append(day_record)
        # Only count toward sync_interval once warmup is over
        if not in_warmup:
            days_since_sync += 1

        # Print day row (always)
        _warmup_tag = " [W]" if day_record["in_warmup"] else "    "

        def _fmt_client_col(n: str, _rec: dict = day_record) -> str:
            n_samples = _rec.get(f"{n}_n", 0)
            if n_samples == 0:
                return f"{'---':>8}  {'---':>8}  {0:>5}"
            return (
                f"{_rec.get(f'{n}_mae', float('nan')):>8.6f}"
                f"  {_rec.get(f'{n}_rmse', float('nan')):>8.6f}"
                f"  {n_samples:>5}"
            )

        _row = "  ".join(_fmt_client_col(n) for n in client_names)
        print(f"  {day_key:<6}{_warmup_tag}  {_row}")

        # ---------------------------------------------------------------
        # Step 3: FL Aggregation
        #   - forced round immediately after warmup ends
        #   - periodic round every sync_interval days (post-warmup)
        # ---------------------------------------------------------------
        _do_aggregate = algorithm != "isolated" and (warmup_ended_today or days_since_sync >= sync_interval)
        if warmup_ended_today and not in_warmup:
            # Reset counter so the next periodic sync starts fresh
            days_since_sync = 0

        if _do_aggregate:
            agg_fn = AGGREGATION_FNS[algorithm]
            state_dicts = [m.state_dict() for m in models]

            # -----------------------------------------------------------------
            # Warmup cache — save pre-aggregation client states so future runs
            # with the same warmup config can skip straight to this point.
            # Saved before aggregation: weights are purely from isolated training
            # and contain no algorithm-specific bias.
            # -----------------------------------------------------------------
            if warmup_ended_today and _use_cache and not _warmup_cache_loaded:
                assert _warmup_cache_file is not None
                _warmup_cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "cache_key": _warmup_cache_key,
                        "warmup_days": warmup_days,
                        "client_names": client_names,
                        "client_states": list(state_dicts),
                        "per_day": list(per_day),
                        "all_preds": [[a.copy() for a in preds] for preds in _all_preds],
                        "all_tgts": [[a.copy() for a in tgts] for tgts in _all_tgts],
                    },
                    _warmup_cache_file,
                )
                print(f"  {'':6}      [cache] Warmup snapshot saved → {_warmup_cache_file}")

            # Capture local-only layers before aggregation
            if algorithm == "fedbn":
                local_bn = [_extract_local_keys(sd, _is_bn_key) for sd in state_dicts]
            if algorithm == "fedper":
                local_head = [_extract_local_keys(sd, _is_head_key) for sd in state_dicts]

            new_global, wc = agg_fn(state_dicts[0], state_dicts)
            last_global_state_dict = copy.deepcopy(new_global)

            # Distribute back — restore local layers per client
            for i, model in enumerate(models):
                merged = new_global
                if algorithm == "fedbn":
                    merged = _restore_local_keys(merged, local_bn[i])
                elif algorithm == "fedper":
                    merged = _restore_local_keys(merged, local_head[i])
                model.load_state_dict(merged)

            round_tag = "[WARMUP END] " if warmup_ended_today else ""
            per_day[-1]["fl_round_after"] = fl_round_idx
            per_round.append(
                {
                    "fl_round": fl_round_idx,
                    "day": day_key,
                    "day_num": day_num,
                    "weight_change": wc,
                    "days_in_round": days_since_sync if not warmup_ended_today else warmup_days,
                    "warmup_round": warmup_ended_today,
                }
            )
            if not warmup_ended_today:
                days_since_sync = 0
            fl_round_idx += 1

            _loss_str = "  ".join(
                f"{_name}={_loss:.5f}"
                for _name, _loss in zip(client_names, losses, strict=False)
                if day_record.get(f"{_name}_n", 0) > 0
            )
            print(
                f"  {'':6}      └─ {round_tag}FL Round {fl_round_idx:3d}  ({per_round[-1]['days_in_round']}d)  ΔW={wc:.5f}  losses: {_loss_str}"
            )

    # -----------------------------------------------------------------------
    # Overall metrics — built from accumulated predictions, no re-inference
    # -----------------------------------------------------------------------
    overall_metrics: dict[str, dict[str, float]] = {}
    for i, name in enumerate(client_names):
        if _all_preds[i]:
            overall_metrics[name] = compute_metrics(np.concatenate(_all_preds[i]), np.concatenate(_all_tgts[i]))
        else:
            overall_metrics[name] = {}

    _nw = max(10, max(len(n) for n in client_names))
    print(f"\n{'='*80}")
    print(f"  Final Results — {algorithm.upper()}  (FL rounds fired: {fl_round_idx})")
    print(f"{'='*80}")
    print(f"  {'Client':<{_nw}}  {'MAE':>9}  {'RMSE':>9}  {'MAPE':>8}  {'WAPE':>8}  {'SMAPE':>8}  {'Count':>8}")
    print(f"  {'-'*_nw}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name in client_names:
        m = overall_metrics[name]
        print(
            f"  {name:<{_nw}}  "
            f"{m.get('mae', float('nan')):>9.5f}  "
            f"{m.get('rmse', float('nan')):>9.5f}  "
            f"{m.get('mape', float('nan')):>8.2%}  "
            f"{m.get('wape', float('nan')):>7.2f}%  "
            f"{m.get('smape', float('nan')):>7.2f}%  "
            f"{m.get('count', 0):>8}"
        )
    print(f"{'='*80}\n")

    return {
        "algorithm": algorithm,
        "sync_interval": sync_interval,
        "num_days": len(all_days),
        "start_day": start_day,
        "warmup_days": warmup_days,
        "warmup_cache_loaded": _warmup_cache_loaded,
        "client_names": client_names,
        "fl_rounds_total": fl_round_idx,
        "per_day": per_day,
        "per_round": per_round,
        "overall_metrics": overall_metrics,
        "models": models,
        "last_global_state_dict": last_global_state_dict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the pretrained model checkpoint (shared starting point for all clients).",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help=(
        "Prepared dataset path. Repeat to add clients "
        "(e.g. --dataset data/folsom_prepared --dataset data/karpicko_prepared). "
        "At least two required. Clients are created in the order given."
    ),
)
@click.option(
    "--algorithm",
    type=click.Choice(["isolated", "fedavg", "fedbn", "fedper"], case_sensitive=False),
    default="fedbn",
    show_default=True,
    help=(
        "FL aggregation algorithm.\n"
        "  isolated — no federation (lower bound).\n"
        "  fedavg   — standard weight averaging.\n"
        "  fedbn    — FedAvg excluding BatchNorm layers.\n"
        "  fedper   — aggregate backbone only, keep head local."
    ),
)
@click.option(
    "--sync-interval",
    type=int,
    default=7,
    show_default=True,
    help="Days between FL aggregation rounds (7 = weekly, 14 = bi-weekly).",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    show_default=True,
    help="Number of days to simulate (-1 = all available days).",
)
@click.option(
    "--start-day",
    type=str,
    default="0101",
    show_default=True,
    help="Start day in MMDD format (e.g. 0101 for January 1).",
)
@click.option(
    "--warmup-days",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Number of initial days where all methods train in isolation (no federation). "
        "At the end of the warmup period a single forced aggregation round fires, "
        "then normal sync-interval scheduling resumes.  0 = disabled."
    ),
)
@click.option(
    "--warmup-cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    show_default=True,
    help=(
        "Directory for caching per-client model states at the end of the warmup phase. "
        "The cache key is derived from all warmup-relevant hyperparameters (checkpoint, "
        "datasets, warmup_days, lr, …) but NOT the algorithm, so the same snapshot is "
        "reused across all algorithm runs that share the same warmup config.  "
        "If the cache file already exists the warmup phase is skipped entirely.  "
        "Pass None (default) to disable caching."
    ),
)
@click.option("--adapt-lr", type=float, default=1e-4, show_default=True, help="Local learning rate (all clients).")
@click.option("--adapt-weight-decay", type=float, default=1e-3, show_default=True, help="AdamW weight decay.")
@click.option(
    "--adapt-max-grad-norm",
    type=float,
    default=1.0,
    show_default=True,
    help="Gradient clipping max norm (-1 = disabled).",
)
@click.option("--adapt-batch-size", type=int, default=32, show_default=True, help="Mini-batch size for local training.")
@click.option(
    "--adapt-epochs",
    type=int,
    default=1,
    show_default=True,
    help="Local training epochs per day (all clients).",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    show_default=True,
    help="Compute device.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
@click.option(
    "--save-results",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Directory to save results.  Creates:\n"
        "  <dir>/per_day.csv     — daily MAE/RMSE for all clients.\n"
        "  <dir>/per_round.csv   — per-FL-round statistics.\n"
        "  <dir>/summary.json    — overall metrics and config."
    ),
)
def main(
    checkpoint: Path,
    dataset: tuple[Path, ...],
    algorithm: str,
    sync_interval: int,
    num_days: int,
    start_day: str,
    warmup_days: int,
    warmup_cache_dir: Path | None,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_batch_size: int,
    adapt_epochs: int,
    device: str,
    seed: int,
    save_results: Path | None,
) -> None:
    """Run the FL benchmark comparing federation algorithms on solar irradiance forecasting."""
    # Build ordered client list from all --dataset flags
    # Strip common suffixes (_prepared) to get a clean site name
    clients: list[tuple[str, Path]] = []
    for ds in dataset:
        raw = ds.parent.name if ds.name == "." else ds.name
        name = raw.removesuffix("_prepared")
        clients.append((name, ds))

    results = run_fl_benchmark(
        checkpoint_path=checkpoint,
        clients=clients,
        algorithm=algorithm,
        sync_interval=sync_interval,
        num_days=num_days,
        start_day=start_day,
        warmup_days=warmup_days,
        warmup_cache_dir=warmup_cache_dir,
        adapt_lr=adapt_lr,
        adapt_weight_decay=adapt_weight_decay,
        adapt_max_grad_norm=adapt_max_grad_norm,
        adapt_batch_size=adapt_batch_size,
        adapt_epochs=adapt_epochs,
        device=device,
        seed=seed,
    )

    if save_results is not None:
        save_results.mkdir(parents=True, exist_ok=True)

        # Per-day CSV
        df_day = pd.DataFrame(results["per_day"])
        df_day.to_csv(save_results / "per_day.csv", index=False)

        # Per-round CSV
        if results["per_round"]:
            df_round = pd.DataFrame(results["per_round"])
            df_round.to_csv(save_results / "per_round.csv", index=False)

        # Per-client model checkpoints (last model state after simulation)
        ckpt_dir = save_results / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        _final_models: list[Forecaster] = results["models"]
        for _name, _model in zip(results["client_names"], _final_models, strict=False):
            _ckpt_path = ckpt_dir / f"{_name}_final.pt"
            torch.save({"state_dict": _model.state_dict(), "algorithm": algorithm}, _ckpt_path)
            print(f"  {_ckpt_path}")
        # Save the last global model broadcast (None for isolated — no aggregation)
        _global_sd = results["last_global_state_dict"]
        if _global_sd is not None:
            _global_path = ckpt_dir / "global_final.pt"
            torch.save({"state_dict": _global_sd, "algorithm": algorithm}, _global_path)
            print(f"  {_global_path}")

        # Summary JSON (exclude large per_day / per_round lists and non-serialisable models)
        summary = {k: v for k, v in results.items() if k not in ("per_day", "per_round", "models", "last_global_state_dict")}
        summary["config"] = {
            "checkpoint": str(checkpoint),
            "clients": [(name, str(path)) for name, path in clients],
            "warmup_days": warmup_days,
            "warmup_cache_dir": str(warmup_cache_dir) if warmup_cache_dir else None,
            "warmup_cache_loaded": results["warmup_cache_loaded"],
            "adapt_lr": adapt_lr,
            "adapt_epochs": adapt_epochs,
            "adapt_weight_decay": adapt_weight_decay,
            "adapt_max_grad_norm": adapt_max_grad_norm,
            "adapt_batch_size": adapt_batch_size,
            "seed": seed,
        }
        with (save_results / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)

        print(f"Results saved to {save_results}/")
        print(f"  per_day.csv   ({len(results['per_day'])} rows)")
        if results["per_round"]:
            print(f"  per_round.csv ({len(results['per_round'])} rows)")
        print("  summary.json")
        _n_global = 1 if results["last_global_state_dict"] is not None else 0
        print(f"  checkpoints/  ({len(_final_models)} local models{', 1 global' if _n_global else ''})")


if __name__ == "__main__":
    main()
