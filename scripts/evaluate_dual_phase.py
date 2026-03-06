"""Evaluate solar irradiance forecasting with Dual-Phase Online Adaptation.

Runs :class:`~cirr.tta.DualPhaseAdapter` in an online day-by-day loop over a
prepared dataset.  This script owns the causal simulation: a FIFO quarantine
queue withholds each ground-truth label for ``horizon_steps`` time steps before
passing it to :meth:`~cirr.tta.DualPhaseAdapter.update`, faithfully reproducing
the real-time lag between prediction and label arrival.

Full data-flow graph (one time step *t*)::

    (image_t, history_t, future_gt_t)
                  │
                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  A  CAUSAL FIREWALL          [this script]                  │
    │     Push {image_t, history_t, gt_t} → causal_queue  (CPU)   │
    │     queue = [ ..., t-H+1, ..., t-1, t ]                     │
    └─────────────────────────────────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │  len(queue) > H ?  │
        └─────────┬──────────┘
     YES │                    │ NO
         ▼                    ▼
    ┌──────────────────┐   (skip B)
    │  B  LAGGED OGD   │   [this script calls adapter.update()]
    │                  │
    │  pop oldest ──►  │
    │  (img_{t-H},     │
    │   hist_{t-H},    │
    │   gt_{t-H})      │
    │                  │
    │  ŷ = base_model( │
    │        img_{t-H},│
    │        hist_{t-H}│
    │      )           │
    │  loss = BerHu(   │
    │    ŷ, gt_{t-H})  │
    │                  │
    │  SGD.step() ──►  │ base_model weights updated  (persists)
    └──────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  C  ZERO-LAG TTA             [adapter.predict()]            │
    │  (ephemeral — base_model never touched)                     │
    │                                                             │
    │  tta_model = deepcopy(base_model)                           │
    │  freeze all  ►  unfreeze BN/LN affine params (γ, β) only    │
    │                                                             │
    │  MEMO:  N augmented views of image_t                        │
    │         loss = Var({ tta_model(aug_i, history_t) })         │
    │                                                             │
    │  NOTE:  buffer of recent (image, history) pairs             │
    │         loss = Var({ tta_model(buf_i,  hist_i ) })          │
    │                                                             │
    │  AdamW.step()    ►  tta_model updated  (base_model frozen)  │
    └─────────────────────────────────────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  D  FINAL INFERENCE          [adapter.predict()]            │
    │     tta_model.eval() + torch.no_grad()                      │
    │     prediction = tta_model(image_t, history_t)              │
    │     del tta_model   ◄── GPU memory freed immediately        │
    └─────────────────────────────────────────────────────────────┘
                  │
                  ▼
            ŷ_{t+H}  returned to caller

Usage examples
--------------
# MEMO-style TTA, default settings (horizon=3, head-only OGD):
python scripts/evaluate_dual_phase.py \\
    --checkpoint checkpoints/epoch_024.ckpt \\
    --dataset-path data/poznan_prepared

# NOTE-style TTA, 5-day window, encoder OGD scope:
python scripts/evaluate_dual_phase.py \\
    --checkpoint checkpoints/epoch_024.ckpt \\
    --dataset-path data/poznan_prepared \\
    --tta-method note --note-buffer-size 32 \\
    --ogd-scope head_and_encoder --num-days 5

# Save per-day report:
python scripts/evaluate_dual_phase.py \\
    --checkpoint checkpoints/epoch_024.ckpt \\
    --dataset-path data/poznan_prepared \\
    --save-results
"""

from collections import deque
from pathlib import Path
import sys
from typing import Any

import click
import numpy as np
import torch
from tqdm import tqdm

from cirr.datamodules.loading import load_dataset
from cirr.evaluation.printing import print_metrics_header, print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.tta import DualPhaseAdapter
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    adapter: DualPhaseAdapter,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
    horizon_steps: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Run the day-by-day online evaluation loop with DualPhaseAdapter.

    This function owns the causal simulation: a ``deque`` queue withholds each
    ground-truth label for ``horizon_steps`` time steps before releasing it to
    :meth:`~cirr.tta.DualPhaseAdapter.update`.  The queue is flushed at every
    day boundary to prevent overnight ground-truth leakage.

    Parameters
    ----------
    adapter : DualPhaseAdapter
        Configured adapter (already moved to ``device``).
    dataset : SkyImageDataset
        Test dataset; items are ``(image, history, target, name)``.
    day_index_map : dict[str, list[int]]
        Maps MMDD keys to dataset indices for that day.
    days_to_eval : list[str]
        Ordered MMDD keys to evaluate.
    device : str
        Target device string (``'cuda'`` or ``'cpu'``).
    horizon_steps : int
        Causal queue depth: labels are withheld for this many steps.

    Returns
    -------
    overall_metrics, per_day_metrics, predictions_data
    """
    all_predictions: list[float] = []
    all_targets: list[float] = []
    predictions_data: list[dict[str, Any]] = []
    per_day_metrics: dict[str, dict[str, float]] = {}
    causal_queue: deque[dict[str, torch.Tensor]] = deque()

    print_metrics_header()

    for day_key in days_to_eval:
        day_indices = day_index_map.get(day_key, [])
        if not day_indices:
            continue

        day_preds: list[float] = []
        day_targets: list[float] = []

        for idx in tqdm(day_indices, desc=f"Day {day_key}", unit="sample", leave=False, file=sys.stderr):
            image_input, irradiance_history, target_irradiance, image_name = dataset[idx]

            images = image_input.unsqueeze(0).to(device)  # [1, C, H, W]
            irr_history = irradiance_history.unsqueeze(0).to(device)  # [1, hist]
            target = target_irradiance.unsqueeze(0).to(device)  # [1, 1]

            if isinstance(image_name, bytes):
                image_name = image_name.decode("utf-8")

            # A: push current sample into the causal queue (kept on CPU).
            causal_queue.append(
                {
                    "image": image_input,  # [C, H, W]
                    "history": irradiance_history,  # [hist_len]
                    "gt": target_irradiance.unsqueeze(0),  # [1, 1]
                }
            )

            # B: pop the oldest sample once the horizon lag has elapsed.
            if len(causal_queue) > horizon_steps:
                oldest = causal_queue.popleft()
                adapter.update(
                    oldest["image"].to(device),
                    oldest["history"].to(device),
                    oldest["gt"].to(device),
                )

            # C+D: TTA-adapt a deepcopy and predict with it.
            mean, _ = adapter.predict(images, irr_history)

            pred_np = mean.detach().cpu().numpy().flatten()
            target_np = target.cpu().numpy().flatten()

            day_preds.extend(pred_np)
            day_targets.extend(target_np)
            all_predictions.extend(pred_np)
            all_targets.extend(target_np)

            predictions_data.append(
                {
                    "timestamp": image_name if isinstance(image_name, str) else None,
                    "day": day_key,
                    "prediction": pred_np[0] if len(pred_np) == 1 else pred_np,
                    "target": target_np[0] if len(target_np) == 1 else target_np,
                    "error": float(pred_np[0] - target_np[0]),
                    "absolute_error": float(abs(pred_np[0] - target_np[0])),
                }
            )

        if day_preds:
            day_metrics = compute_metrics(np.array(day_preds), np.array(day_targets))
            per_day_metrics[day_key] = day_metrics
            print(
                f"{day_key:<6} {day_metrics['mae']:<10.4f} {day_metrics['rmse']:<10.4f} "
                f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% {day_metrics['count']:<8}",
                flush=True,
            )

        # Flush the causal queue at day boundaries to prevent overnight
        # ground-truth leakage.
        causal_queue.clear()

    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    return overall_metrics, per_day_metrics, predictions_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint file.",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to prepared dataset directory (e.g. data/poznan_prepared).",
)
@click.option(
    "--tta-method",
    type=click.Choice(["memo", "note", "none"], case_sensitive=False),
    default="memo",
    show_default=True,
    help="Unsupervised TTA variant. 'none' = OGD-only (no ephemeral TTA).",
)
@click.option(
    "--horizon-steps",
    type=int,
    default=15,
    show_default=True,
    help="Causal queue depth: time steps between prediction and label arrival. ",
)
@click.option(
    "--ogd-lr",
    type=float,
    default=1e-5,
    show_default=True,
    help="Learning rate for the persistent supervised OGD optimizer (SGD).",
)
@click.option(
    "--ogd-scope",
    type=click.Choice(["head_only", "head_and_encoder"], case_sensitive=False),
    default="head_only",
    show_default=True,
    help="Parameters updated by OGD: regression head only, or head + last encoder block.",
)
@click.option(
    "--ogd-loss",
    type=click.Choice(["berhu", "huber", "mse"], case_sensitive=False),
    default="berhu",
    show_default=True,
    help="Supervised loss function used in the OGD phase.",
)
@click.option(
    "--lr-tta",
    type=float,
    default=1e-5,
    show_default=True,
    help="Learning rate for the ephemeral TTA AdamW optimizer (norm-layer affines).",
)
@click.option(
    "--n-augmentations",
    type=int,
    default=8,
    show_default=True,
    help="(MEMO) Number of augmented image views used for variance minimisation.",
)
@click.option(
    "--niter",
    type=int,
    default=1,
    show_default=True,
    help="Number of TTA gradient steps per time step.",
)
@click.option(
    "--note-buffer-size",
    type=int,
    default=64,
    show_default=True,
    help="(NOTE) Maximum number of recent samples retained in the persistent buffer.",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    show_default=True,
    help="Number of days to evaluate. -1 means all available days.",
)
@click.option(
    "--start-day",
    type=str,
    default="0101",
    show_default=True,
    help="Starting day in MMDD format (e.g. 0101).",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    show_default=True,
    help="Device to run on ('cuda' or 'cpu').",
)
@click.option(
    "--save-results",
    is_flag=True,
    default=False,
    help="Save per-day metrics report and predictions CSV to disk.",
)
def main(
    checkpoint: Path,
    dataset_path: Path,
    tta_method: str,
    horizon_steps: int,
    ogd_lr: float,
    ogd_scope: str,
    ogd_loss: str,
    lr_tta: float,
    n_augmentations: int,
    niter: int,
    note_buffer_size: int,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
) -> None:
    """Evaluate solar irradiance forecasting with Dual-Phase Online Adaptation."""

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    dataset_name = dataset_path.name.split("_")[0]
    method_tag = f"dual_{tta_method}_{ogd_scope}"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {checkpoint}...")
    model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    if hasattr(model.image_encoder, "fc") and not isinstance(model.image_encoder.fc, torch.nn.Identity):
        model.image_encoder.fc = torch.nn.Identity()

    # ------------------------------------------------------------------
    # Build adapter
    # ------------------------------------------------------------------
    print(
        f"Building DualPhaseAdapter ("
        f"tta_method={tta_method!r}, horizon_steps={horizon_steps}, "
        f"ogd_lr={ogd_lr}, ogd_scope={ogd_scope!r}, ogd_loss={ogd_loss!r}, "
        f"lr_tta={lr_tta}, n_augmentations={n_augmentations}, niter={niter}, "
        f"note_buffer_size={note_buffer_size})..."
    )
    adapter = DualPhaseAdapter(
        base_model=model,
        lr_ogd=ogd_lr,
        lr_tta=lr_tta,
        tta_method=tta_method,
        ogd_scope=ogd_scope,
        ogd_loss=ogd_loss,
        tta_weight_decay=0.0,
        n_augmentations=n_augmentations,
        niter_tta=niter,
        note_buffer_size=note_buffer_size,
    ).to(device)

    # ------------------------------------------------------------------
    # Load dataset and determine days to evaluate
    # ------------------------------------------------------------------
    print(f"Loading dataset from {dataset_path}...")
    test_dataset = load_dataset(dataset_path, split="test")
    day_index_map = group_indices_by_day(test_dataset._periods)
    print(f"Test set size: {len(test_dataset)} samples")
    print(f"Number of unique days: {len(day_index_map)}\n")

    all_days = generate_year_days()
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day: {start_day!r}")
    start_idx = all_days.index(start_day)

    days_to_eval = all_days[start_idx : start_idx + num_days] if num_days > 0 else all_days[start_idx:]
    days_to_eval = [d for d in days_to_eval if d in day_index_map]

    total_samples = sum(len(day_index_map[d]) for d in days_to_eval)
    print(f"Evaluating {len(days_to_eval)} days starting from {start_day}")
    print(f"Total samples: {total_samples}\n")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print("Running Dual-Phase Online Adaptation evaluation...\n")

    overall_metrics, per_day_metrics, predictions_data = evaluate(
        adapter=adapter,
        dataset=test_dataset,
        day_index_map=day_index_map,
        days_to_eval=days_to_eval,
        device=device,
        horizon_steps=horizon_steps,
    )

    print_overall_metrics(overall_metrics, tag=f"{dataset_name}  [{method_tag}]")

    if save_results:
        csv_path = f"{dataset_name}_{method_tag}_predictions.csv"
        txt_path = f"{dataset_name}_{method_tag}_results.txt"
        save_predictions_csv(predictions_data, csv_path)
        print(f"\nPredictions saved to {csv_path}")
        extra = [
            f"TTA: DualPhaseAdapter  tta_method={tta_method}"
            f"  horizon_steps={horizon_steps}  ogd_lr={ogd_lr}  ogd_scope={ogd_scope}"
            f"  ogd_loss={ogd_loss}  lr_tta={lr_tta}  n_aug={n_augmentations}"
            f"  niter={niter}  note_buffer_size={note_buffer_size}"
        ]
        save_metrics_report(
            tag=method_tag,
            dataset_name=dataset_name,
            checkpoint=checkpoint,
            dataset_path=dataset_path,
            overall_metrics=overall_metrics,
            per_day_metrics=per_day_metrics,
            path=txt_path,
            extra_lines=extra,
        )
        print(f"Results saved to {txt_path}")


if __name__ == "__main__":
    main()
