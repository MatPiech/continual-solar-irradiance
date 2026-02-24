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
  fedprox   FedAvg + proximal regularization (µ/2)||w - w_global||² in local
            loss.  Prevents clients from drifting far from the global model.
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
    model = Forecaster(
        model_name="timm-resnet50",
        pretrained=False,
        input_channels=3,
        loss_function="GNLL",
        lr=1e-3,
        lr_patience=5,
    ).to(device)
    data = torch.load(checkpoint_path, map_location=device)
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
    proximal_anchor: dict[str, torch.Tensor] | None,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    proximal_mu: float,
    batch_size: int,
    epochs: int,
    device: str,
) -> float:
    """Train *model* for *epochs* epochs on *day_indices*.

    For FedProx the caller passes ``proximal_mu > 0`` and a
    ``proximal_anchor`` dict that holds the trainable parameter snapshots
    from *the start of the current FL round* (i.e. the last server broadcast).
    The proximal loss is:

        loss_total = loss_gnll + (µ/2) * ||w - w_anchor||²

    Critically the anchor is fixed for the entire sync window (7 days),
    not re-snapshotted on each day — this matches the FedProx paper.

    For plain FedAvg / FedBN / FedPer pass ``proximal_mu=0.0``.

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

    # For FedProx use the caller-supplied anchor (fixed to round-start server
    # weights); fall back to None which disables the proximal term.
    proximal_ref = proximal_anchor if (proximal_mu > 0.0 and proximal_anchor is not None) else None

    total_loss = 0.0
    count = 0

    for _ in range(epochs):
        for batch in batches:
            imgs = torch.stack([dataset[idx][0] for _, idx in batch]).to(device)
            hist = torch.stack([dataset[idx][1] for _, idx in batch]).to(device)
            tgts = torch.stack([dataset[idx][2] for _, idx in batch]).to(device)

            optimizer.zero_grad()
            mean, var = model(imgs, hist)
            base_loss = model.loss_fn(mean, tgts, var) if var is not None else model.loss_fn(mean, tgts)

            if proximal_mu > 0.0 and proximal_ref:
                prox_terms = [
                    torch.mean((p - proximal_ref[n]) ** 2)
                    for n, p in model.named_parameters()
                    if p.requires_grad and n in proximal_ref
                ]
                prox = torch.stack(prox_terms).mean() if prox_terms else torch.tensor(0.0, device=device)
                loss = base_loss + (proximal_mu / 2.0) * prox
            else:
                loss = base_loss

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
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_fl_benchmark(
    checkpoint_path: Path,
    clients: list[tuple[str, Path]],
    algorithm: str,
    sync_interval: int,
    num_days: int,
    start_day: str,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    fedprox_mu: float,
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
    algorithm           : One of "isolated", "fedavg", "fedprox", "fedbn", "fedper".
    sync_interval       : Days between FL aggregation rounds (e.g. 7 or 14).
    num_days            : Total days to simulate (-1 = all).
    start_day           : MMDD string (e.g. "0101").
    adapt_lr            : Local learning rate (all clients).
    adapt_weight_decay  : AdamW weight decay.
    adapt_max_grad_norm : Gradient clipping (-1 = disabled).
    fedprox_mu          : Proximal coefficient µ (fedprox only).
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
    print(f"  FL Benchmark — algorithm: {algorithm.upper()}  sync_interval: {sync_interval} days")
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
    _nw = max(10, max(len(n) for n in client_names))
    _header_names = "  ".join(f"{n:^{_nw + 32}s}" for n in client_names)
    _header_metrics = "  ".join(f"{'MAE':>9}  {'RMSE':>9}  {'N':>6}  {'Loss':>9}" for _ in client_names)
    _sep = "  ".join(["-" * (_nw + 32)] * len(client_names))
    print(f"  {'Day':<6}  {_header_names}")
    print(f"  {'':6}  {_header_metrics}")
    print(f"  {'------'}  {_sep}")

    # --- Per-client state for FedBN / FedPer ---
    local_bn: list[dict[str, Any]] = [{} for _ in clients]
    local_head: list[dict[str, Any]] = [{} for _ in clients]

    # --- FedProx: anchors fixed to server weights at round start ---
    mu = fedprox_mu if algorithm == "fedprox" else 0.0

    def _make_proximal_anchor(model: Forecaster) -> dict[str, torch.Tensor] | None:
        if algorithm != "fedprox":
            return None
        return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    prox_anchors: list[dict[str, torch.Tensor] | None] = [_make_proximal_anchor(m) for m in models]

    per_day: list[dict[str, Any]] = []
    per_round: list[dict[str, Any]] = []
    days_since_sync = 0
    fl_round_idx = 0

    # -----------------------------------------------------------------------
    # Main loop — identical treatment for every client
    # -----------------------------------------------------------------------
    for day_num, day_key in enumerate(all_days):
        day_record: dict[str, Any] = {"day": day_key, "day_num": day_num, "fl_round_after": None}
        losses: list[float] = []

        for i, (name, model) in enumerate(zip(client_names, models, strict=False)):
            indices = day_maps[i].get(day_key, [])
            day_record[f"{name}_n"] = len(indices)

            # Step 1: Predict
            preds, tgts = _predict_day(model, datasets[i], indices, device)
            metrics = compute_metrics(preds, tgts) if len(preds) > 0 else {}
            day_record[f"{name}_mae"] = metrics.get("mae", float("nan"))
            day_record[f"{name}_rmse"] = metrics.get("rmse", float("nan"))

            # Step 2: Train — same loop for all clients
            loss = _train_day(
                model,
                datasets[i],
                indices,
                proximal_anchor=prox_anchors[i],
                lr=adapt_lr,
                weight_decay=adapt_weight_decay,
                max_grad_norm=adapt_max_grad_norm,
                proximal_mu=mu,
                batch_size=adapt_batch_size,
                epochs=adapt_epochs,
                device=device,
            )
            losses.append(loss)

        per_day.append(day_record)
        days_since_sync += 1

        # Print day row (always)
        _row = "  ".join(
            f"{day_record.get(f'{n}_mae', float('nan')):>9.5f}"
            f"  {day_record.get(f'{n}_rmse', float('nan')):>9.5f}"
            f"  {day_record.get(f'{n}_n', 0):>6}"
            f"  {losses[j]:>9.5f}"
            for j, n in enumerate(client_names)
        )
        print(f"  {day_key:<6}  {_row}")

        # Step 3: FL Aggregation Round (every sync_interval days)
        if algorithm != "isolated" and days_since_sync >= sync_interval:
            agg_fn = AGGREGATION_FNS[algorithm]
            state_dicts = [m.state_dict() for m in models]

            # Capture local-only layers before aggregation
            if algorithm == "fedbn":
                local_bn = [_extract_local_keys(sd, _is_bn_key) for sd in state_dicts]
            if algorithm == "fedper":
                local_head = [_extract_local_keys(sd, _is_head_key) for sd in state_dicts]

            new_global, wc = agg_fn(state_dicts[0], state_dicts)

            # Distribute back — restore local layers per client
            for i, model in enumerate(models):
                merged = new_global
                if algorithm == "fedbn":
                    merged = _restore_local_keys(merged, local_bn[i])
                elif algorithm == "fedper":
                    merged = _restore_local_keys(merged, local_head[i])
                model.load_state_dict(merged)

            # Refresh FedProx anchors to newly broadcast weights
            prox_anchors = [_make_proximal_anchor(m) for m in models]

            per_day[-1]["fl_round_after"] = fl_round_idx
            per_round.append(
                {
                    "fl_round": fl_round_idx,
                    "day": day_key,
                    "day_num": day_num,
                    "weight_change": wc,
                    "days_in_round": days_since_sync,
                }
            )
            days_since_sync = 0
            fl_round_idx += 1

            _loss_str = "  ".join(f"{_name}={_loss:.5f}" for _name, _loss in zip(client_names, losses, strict=False))
            print(
                f"  {'':6}  └─ FL Round {fl_round_idx:3d}  ({per_round[-1]['days_in_round']}d)  ΔW={wc:.5f}  losses: {_loss_str}"
            )

    # -----------------------------------------------------------------------
    # Overall metrics (final model state)
    # -----------------------------------------------------------------------
    overall_metrics: dict[str, dict[str, float]] = {}
    for i, name in enumerate(client_names):
        all_preds, all_tgts = [], []
        for d in per_day:
            indices = day_maps[i].get(d["day"], [])
            if indices:
                p, t = _predict_day(models[i], datasets[i], indices, device)
                all_preds.append(p)
                all_tgts.append(t)
        overall_metrics[name] = compute_metrics(np.concatenate(all_preds), np.concatenate(all_tgts)) if all_preds else {}

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
        "client_names": client_names,
        "fl_rounds_total": fl_round_idx,
        "per_day": per_day,
        "per_round": per_round,
        "overall_metrics": overall_metrics,
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
    type=click.Choice(["isolated", "fedavg", "fedprox", "fedbn", "fedper"], case_sensitive=False),
    default="fedbn",
    show_default=True,
    help=(
        "FL aggregation algorithm.\n"
        "  isolated — no federation (lower bound).\n"
        "  fedavg   — standard weight averaging.\n"
        "  fedprox  — FedAvg + proximal term in local loss.\n"
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
@click.option("--adapt-lr", type=float, default=1e-4, show_default=True, help="Local learning rate (all clients).")
@click.option("--adapt-weight-decay", type=float, default=1e-3, show_default=True, help="AdamW weight decay.")
@click.option(
    "--adapt-max-grad-norm",
    type=float,
    default=1.0,
    show_default=True,
    help="Gradient clipping max norm (-1 = disabled).",
)
@click.option(
    "--fedprox-mu",
    type=float,
    default=0.01,
    show_default=True,
    help="FedProx proximal coefficient µ (only active for --algorithm fedprox).",
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
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    fedprox_mu: float,
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
        adapt_lr=adapt_lr,
        adapt_weight_decay=adapt_weight_decay,
        adapt_max_grad_norm=adapt_max_grad_norm,
        fedprox_mu=fedprox_mu,
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

        # Summary JSON (exclude large per_day / per_round lists)
        summary = {k: v for k, v in results.items() if k not in ("per_day", "per_round")}
        summary["config"] = {
            "checkpoint": str(checkpoint),
            "clients": [(name, str(path)) for name, path in clients],
            "adapt_lr": adapt_lr,
            "adapt_epochs": adapt_epochs,
            "adapt_weight_decay": adapt_weight_decay,
            "adapt_max_grad_norm": adapt_max_grad_norm,
            "fedprox_mu": fedprox_mu,
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


if __name__ == "__main__":
    main()
