"""Evaluate forecasting with closed-form Sherman-Morrison RLS head adaptation.

Online, one-sample-at-a-time head update using the Sherman-Morrison rank-1
formula — no gradient computation, no optimizer state.

Algorithm (same causal delay structure as Algorithm 1 from the paper):

    For each timestep t:
      1. Predict y_hat_t with the current model (no gradient).
      2. Enqueue the sample in a causal delay buffer of depth ``horizon_steps``.
            3. When the delayed ground-truth arrives, update the model on every
                 sample using a fixed scope chosen via CLI:
                     - ``last_layer`` -> update output layer only
                     - ``full_head``  -> update both linear layers in the head

The head is adapted with a single closed-form RLS step per sample:

    P_{t+1} = (1/λ) * (P_t  -  P_t φ φ^T P_t / (λ + φ^T P_t φ))
    w_{t+1} = w_t  +  P_{t+1} φ * e_t

where φ is the feature vector entering the layer, e_t = y_t - ŷ_t is the
prediction error, and λ ∈ (0, 1] is a forgetting factor.
"""

from collections import deque
from pathlib import Path
import sys
from typing import Any, Literal

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cirr.datamodules.loading import load_dataset
from cirr.evaluation.printing import print_metrics_header, print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Sherman-Morrison closed-form RLS head updater
# ---------------------------------------------------------------------------


class ShermanMorrisonHeadUpdater:
    """Online RLS updater for the forecasting head via Sherman-Morrison formula.

    The forecasting head is assumed to be::

        network_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),   # head[0]  — hidden layer
            nn.ReLU(inplace=True),         # head[1]
            nn.Linear(256, out_dim),       # head[2]  — output layer
        )

    ``scope='last_layer'``
        Only ``head[2]`` (``nn.Linear(256, out_dim)``) is updated.
        Feature vector: ``φ = [a1; 1]`` where a1 = ReLU(W1 x + b1).

    ``scope='full_head'``
        Both linear layers are updated in a single pass:

        * **Output layer** (W2, b2): same as *last_layer*.
        * **Hidden layer** (W1, b1): error credit is propagated back through
          W2 and the ReLU gate, with feature vector ``φ = [x_fusion; 1]``.
          Each of the 256 output units of W1 is treated as an independent
          scalar regression problem sharing the same inverse-covariance
          matrix P1 (standard multi-output RLS).

    Parameters
    ----------
    model : Forecaster
        The model whose head will be adapted (weights updated in-place).
    lambda_init : float
        Initialises ``P = (1/lambda_init) * I``.  Larger ⟹ more
        regularisation, smaller initial Kalman gain.
    forgetting_factor : float
        Exponential forgetting ``λ ∈ (0, 1]``.  1.0 = standard RLS (no
        forgetting); < 1 discounts older data exponentially.
    device : str
        Device for the P matrices (should match the model device).
    """

    def __init__(
        self,
        model: Forecaster,
        lambda_init: float = 1.0,
        forgetting_factor: float = 1.0,
        full_head_regularization_mult: float = 100.0,
        error_clip: float = 0.2,
        device: str = "cuda",
    ) -> None:
        self.lam = forgetting_factor
        self.device = device
        self.error_clip = float(error_clip)

        head = model.network_head
        self._last_linear: nn.Linear = head[2]
        self._first_linear: nn.Linear = head[0]

        self._use_bias_last = self._last_linear.bias is not None
        self._use_bias_first = self._first_linear.bias is not None

        d2 = self._last_linear.in_features  # 256
        d1 = self._first_linear.in_features  # fusion_input_dim

        aug_d2 = d2 + int(self._use_bias_last)
        aug_d1 = d1 + int(self._use_bias_first)

        self._P2 = torch.eye(aug_d2, dtype=torch.float32, device=device) / lambda_init
        self._P1 = torch.eye(aug_d1, dtype=torch.float32, device=device) / (lambda_init * full_head_regularization_mult)

    # ------------------------------------------------------------------

    def _sm_update(self, p: torch.Tensor, phi: torch.Tensor, error: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Sherman-Morrison rank-1 RLS update.

        Returns updated P and Kalman gain vector k.

        ``P_new = (1/λ) * (P - P φ φᵀ P / (λ + φᵀ P φ))``
        ``k = P φ / (λ + φᵀ P φ)``
        """
        # Normalize feature vector to avoid gain spikes on large activations.
        phi_norm = float(torch.linalg.norm(phi).item())
        phi = phi / max(phi_norm, 1e-6)

        p_phi = p @ phi  # [d]
        denom = self.lam + float((phi @ p_phi).item())
        p_new = (1.0 / self.lam) * (p - torch.outer(p_phi, p_phi) / denom)
        k = p_phi / denom
        return p_new, k

    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        model: Forecaster,
        image: torch.Tensor,  # [C, H, W] — CPU tensor from causal queue
        history: torch.Tensor,  # [hist_len] — CPU tensor
        gt: torch.Tensor,  # [1, 1]     — CPU tensor
        device: str,
        scope: Literal["last_layer", "full_head"],
    ) -> None:
        """One closed-form SM-RLS update step.

        For ``scope='last_layer'`` only the output linear layer is updated.
        For ``scope='full_head'`` the hidden linear layer is also updated
        using credit propagated through W2 and the ReLU activation mask.
        """
        imgs = image.unsqueeze(0).to(device)
        hist = history.unsqueeze(0).to(device)
        target = gt.to(device).float()

        # ---- Forward pass: collect activations ----
        # Use the same encoder path as model.forward to stay consistent.
        img_feat = model._encode_image_features(imgs, hist)  # [1, num_features]
        x_fusion = torch.cat([img_feat, hist], dim=-1)  # [1, fusion_dim]

        head = model.network_head
        h1 = head[0](x_fusion)  # [1, 256] pre-ReLU
        relu_mask = (h1 > 0).squeeze(0)  # [256] bool mask
        a1 = head[1](h1)  # [1, 256] post-ReLU
        z = head[2](a1)  # [1, out_dim] pre-softplus
        y_hat = F.softplus(z)  # [1, out_dim] matches model.forward regression output

        error = float((target - y_hat).view(-1)[0].item())
        # Limit one-step correction magnitude to keep online adaptation stable.
        error = float(np.clip(error, -self.error_clip, self.error_clip))

        # Local Jacobian dy/dz for softplus; used for linearized RLS update.
        gain = float(torch.sigmoid(z).view(-1)[0].item())
        gain = max(gain, 1e-4)

        # ---- Update output layer (W2, b2) ----
        a1_flat = a1.squeeze(0).to(self.device)
        phi2 = torch.cat([a1_flat, a1_flat.new_ones(1)]) if self._use_bias_last else a1_flat
        phi2_eff = gain * phi2

        self._P2, k2 = self._sm_update(self._P2, phi2_eff, error)

        # Keep a pre-update snapshot for hidden-layer credit propagation.
        w2_pre = self._last_linear.weight.detach().clone().to(self.device)

        if self._use_bias_last:
            self._last_linear.weight.add_((k2[:-1] * error).unsqueeze(0).to(self._last_linear.weight.device))
            self._last_linear.bias.add_((k2[-1] * error).to(self._last_linear.bias.device))
        else:
            self._last_linear.weight.add_((k2 * error).unsqueeze(0).to(self._last_linear.weight.device))

        if scope == "last_layer":
            return

        # ---- Update hidden layer (W1, b1) via credit propagation ----
        x_flat = x_fusion.squeeze(0).to(self.device)
        phi1 = torch.cat([x_flat, x_flat.new_ones(1)]) if self._use_bias_first else x_flat

        self._P1, k1 = self._sm_update(self._P1, phi1, error)

        # Hidden credit uses local linearization through softplus and ReLU.
        # For scalar output: credit_i ≈ w2_i * (error * gain) * relu_mask_i.
        # Use W2 before its own update for a cleaner local linearization.
        credit = w2_pre[0] * (error * gain) * relu_mask.float().to(self.device)  # [256]

        if self._use_bias_first:
            delta_w1 = torch.outer(credit, k1[:-1])  # [256, fusion_dim]
            self._first_linear.weight.add_(delta_w1.to(self._first_linear.weight.device))
            self._first_linear.bias.add_((credit * k1[-1]).to(self._first_linear.bias.device))
        else:
            self._first_linear.weight.add_(torch.outer(credit, k1).to(self._first_linear.weight.device))


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    model: Forecaster,
    updater: ShermanMorrisonHeadUpdater,
    update_scope: Literal["last_layer", "full_head"],
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
    horizon_steps: int,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
    dict[str, dict[str, int]],
    dict[str, int],
]:
    all_predictions: list[float] = []
    all_targets: list[float] = []
    predictions_data: list[dict[str, Any]] = []
    per_day_metrics: dict[str, dict[str, float]] = {}
    per_day_adaptation: dict[str, dict[str, int]] = {}
    causal_queue: deque[dict[str, Any]] = deque()

    update_stats: dict[str, int] = {
        "considered": 0,
        "updated": 0,
    }

    print_metrics_header()

    for day_key in days_to_eval:
        day_indices = day_index_map.get(day_key, [])
        if not day_indices:
            continue

        day_preds: list[float] = []
        day_targets: list[float] = []
        day_update_stats: dict[str, int] = {
            "considered": 0,
            "updated": 0,
        }

        for idx in tqdm(day_indices, desc=f"Day {day_key}", unit="sample", leave=False, file=sys.stderr):
            image_input, irradiance_history, target_irradiance, image_name = dataset[idx]

            if isinstance(image_name, bytes):
                image_name = image_name.decode("utf-8")

            images = image_input.unsqueeze(0).to(device)
            irr_history = irradiance_history.unsqueeze(0).to(device)
            target = target_irradiance.unsqueeze(0).to(device)

            # --- Inference (no gradient) ---
            model.eval()
            with torch.no_grad():
                output = model(images, irr_history)
                mean_raw = output[0] if isinstance(output, tuple) else output
            mean = torch.clamp(mean_raw, min=0.0)

            pred_np = mean.detach().cpu().numpy().flatten()
            target_np = target.detach().cpu().numpy().flatten()

            day_preds.extend(pred_np)
            day_targets.extend(target_np)
            all_predictions.extend(pred_np)
            all_targets.extend(target_np)

            pred_row_idx = len(predictions_data)
            predictions_data.append(
                {
                    "timestamp": image_name if isinstance(image_name, str) else None,
                    "day": day_key,
                    "prediction": pred_np[0] if len(pred_np) == 1 else pred_np,
                    "target": target_np[0] if len(target_np) == 1 else target_np,
                    "error": float(pred_np[0] - target_np[0]),
                    "absolute_error": float(abs(pred_np[0] - target_np[0])),
                    "update_considered": False,
                    "adapted": False,
                    "adaptation_scope": "not_available",
                    "delayed_abs_error": np.nan,
                }
            )

            # --- Enqueue ---
            causal_queue.append(
                {
                    "image": image_input,
                    "history": irradiance_history,
                    "gt": target_irradiance.unsqueeze(0),
                    "raw_pred": float(mean_raw.detach().cpu().view(-1)[0].item()),
                    "pred_row_idx": pred_row_idx,
                }
            )

            # --- Dequeue oldest and apply SM-RLS update ---
            if len(causal_queue) > horizon_steps:
                oldest = causal_queue.popleft()
                update_stats["considered"] += 1
                day_update_stats["considered"] += 1

                delayed_gt = float(oldest["gt"].view(-1)[0].item())
                delayed_abs_error = abs(float(oldest["raw_pred"]) - delayed_gt)

                oldest_row = predictions_data[oldest["pred_row_idx"]]
                oldest_row["update_considered"] = True
                oldest_row["delayed_abs_error"] = delayed_abs_error

                updater.update(
                    model=model,
                    image=oldest["image"],
                    history=oldest["history"],
                    gt=oldest["gt"],
                    device=device,
                    scope=update_scope,
                )
                update_stats["updated"] += 1
                day_update_stats["updated"] += 1
                oldest_row["adapted"] = True
                oldest_row["adaptation_scope"] = update_scope

        if day_preds:
            day_metrics = compute_metrics(np.array(day_preds), np.array(day_targets))
            per_day_metrics[day_key] = day_metrics
            per_day_adaptation[day_key] = day_update_stats
            print(
                f"{day_key:<6} {day_metrics['mae']:<10.4f} {day_metrics['rmse']:<10.4f} "
                f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% {day_metrics['count']:<16}",
                flush=True,
            )
            print(
                f"       adapt: considered={day_update_stats['considered']} "
                f"updated={day_update_stats['updated']} "
                f"scope={update_scope}",
                flush=True,
            )

        # Prevent overnight ground-truth leakage across day boundaries.
        causal_queue.clear()

    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    print(
        f"\nSM-RLS update stats:"
        f"  considered={update_stats['considered']}"
        f"  updated={update_stats['updated']}"
        f"  scope={update_scope}",
        flush=True,
    )

    return overall_metrics, per_day_metrics, predictions_data, per_day_adaptation, update_stats


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
    "--horizon-steps",
    type=click.IntRange(min=1),
    default=15,
    show_default=True,
    help="Causal delay in samples between prediction and label availability.",
)
@click.option(
    "--update-scope",
    type=click.Choice(["last_layer", "full_head"], case_sensitive=False),
    default="last_layer",
    show_default=True,
    help="SM-RLS update scope applied to every delayed sample.",
)
@click.option(
    "--lambda-init",
    type=float,
    default=1.0,
    show_default=True,
    help="Initialises P = (1/lambda_init)*I.  Larger ⟹ stronger regularisation.",
)
@click.option(
    "--forgetting-factor",
    type=click.FloatRange(min=1e-6, max=1.0),
    default=1.0,
    show_default=True,
    help="Exponential forgetting factor λ ∈ (0, 1]. 1.0 = no forgetting (standard RLS).",
)
@click.option(
    "--full-head-regularization-mult",
    type=click.FloatRange(min=1.0),
    default=100.0,
    show_default=True,
    help="Extra regularization factor for full-head hidden-layer RLS (larger = more conservative).",
)
@click.option(
    "--error-clip",
    type=click.FloatRange(min=1e-6),
    default=0.2,
    show_default=True,
    help="Absolute clip applied to per-sample error before SM-RLS update.",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    show_default=True,
    help="Number of days to evaluate. -1 = all available days.",
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
    horizon_steps: int,
    update_scope: str,
    lambda_init: float,
    forgetting_factor: float,
    full_head_regularization_mult: float,
    error_clip: float,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
) -> None:
    """Evaluate forecasting with closed-form Sherman-Morrison RLS head adaptation."""

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    dataset_name = dataset_path.name.split("_")[0]
    method_tag = "sm_rls"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {checkpoint}...")
    model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    if hasattr(model.image_encoder, "fc") and not isinstance(model.image_encoder.fc, torch.nn.Identity):
        model.image_encoder.fc = torch.nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Build SM-RLS updater
    # ------------------------------------------------------------------
    updater = ShermanMorrisonHeadUpdater(
        model=model,
        lambda_init=lambda_init,
        forgetting_factor=forgetting_factor,
        full_head_regularization_mult=full_head_regularization_mult,
        error_clip=error_clip,
        device=device,
    )

    print(
        f"SM-RLS:  update_scope={update_scope}"
        f"  lambda_init={lambda_init}  forgetting={forgetting_factor}"
        f"  full_head_reg_mult={full_head_regularization_mult}"
        f"  error_clip={error_clip}"
        f"  horizon_steps={horizon_steps}"
    )

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
    print("Running SM-RLS evaluation...\n")
    overall_metrics, per_day_metrics, predictions_data, per_day_adaptation, update_stats = evaluate(
        model=model,
        updater=updater,
        update_scope=update_scope,  # type: ignore[arg-type]
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
            "Method: Sherman-Morrison RLS",
            f"horizon_steps={horizon_steps}",
            f"update_scope={update_scope}",
            f"lambda_init={lambda_init}",
            f"forgetting_factor={forgetting_factor}",
            f"full_head_regularization_mult={full_head_regularization_mult}",
            f"error_clip={error_clip}",
            "Adaptation summary:",
            f"considered={update_stats['considered']}",
            f"updated={update_stats['updated']}",
            f"scope={update_scope}",
            "Per-day adaptation:",
        ]
        for day_key in sorted(per_day_adaptation.keys()):
            s = per_day_adaptation[day_key]
            extra.append(f"{day_key}: considered={s['considered']} updated={s['updated']} " f"scope={update_scope}")
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
