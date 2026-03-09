"""Dual-Phase Online Adaptation — OGD-supervised + TTA-calibrated forecasting.

This module provides :class:`DualPhaseAdapter`, the core Dual-Phase online
adaptation implementation.  The adapter combines a persistent base model
updated via online gradient descent (OGD) with ephemeral zero-lag test-time
adaptation (TTA) at inference time.

The adapter exposes two public methods:

* :meth:`update` — Phase B: one supervised OGD gradient step on ``base_model``
  using a ``(image, history, gt)`` triple whose label has already been observed.
* :meth:`predict` — Phases C+D: creates an ephemeral deepcopy of ``base_model``,
  adapts its normalisation-layer affines via MEMO or NOTE, then returns the
  prediction; the copy is discarded immediately.

The causal firewall simulation (FIFO quarantine queue, horizon_steps) is the
responsibility of the evaluation driver (``evaluate_dual_phase.py``), keeping
this class free of simulation-specific concerns.  See that script's module
docstring for the full A→B→C→D data-flow diagram.

Design properties
-----------------
**Ephemeral TTA**
    Phase C uses a ``copy.deepcopy(base_model)`` created and discarded each
    step.  TTA gradients therefore **never accumulate** in ``base_model``;
    only Phase B (OGD) persistently updates model weights.

**Pluggable unsupervised TTA**
    ``tta_method='memo'`` — per-sample augmented-view variance minimisation
    (episodic, no memory).
    ``tta_method='note'`` — variance minimisation on a persistent FIFO buffer
    of recent ``(image, history)`` pairs, giving richer BN batch statistics.
    ``tta_method='none'`` — skip TTA entirely; ``predict()`` runs a plain
    eval forward pass on ``base_model``.  Useful baseline to isolate whether
    OGD alone helps or hurts.

Architecture assumptions
------------------------
The wrapped ``base_model`` must expose:

* ``base_model.network_head``   — ``nn.Sequential`` regression head
* ``base_model.image_encoder``  — timm image encoder with ``.children()``

These are satisfied by :class:`cirr.models.forecaster.Forecaster`.

References
----------
.. [1] Zhang et al. "MEMO: Test Time Robustness via Adaptation and
       Augmentation." NeurIPS 2022. https://arxiv.org/abs/2110.09506
.. [2] Gong et al. "NOTE: Robust Continual Test-time Adaptation Against
       Temporal Correlation." NeurIPS 2022. https://arxiv.org/abs/2208.05117
"""

from collections import deque
from copy import deepcopy
from typing import Literal

import torch
import torch.nn as nn

from cirr.losses import BerHuLoss
from cirr.tta.memo_reg import _random_augment, prediction_variance_loss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of samples in the NOTE buffer before TTA adaptation is
# triggered; below this threshold BN batch statistics are unreliable.
_MIN_NOTE_BUFFER: int = 4


# ---------------------------------------------------------------------------
# DualPhaseAdapter
# ---------------------------------------------------------------------------


class DualPhaseAdapter(nn.Module):
    """Dual-Phase Online Adapter: persistent OGD + ephemeral TTA.

    Parameters
    ----------
    base_model : nn.Module
        A :class:`~cirr.models.forecaster.Forecaster` instance (or any model
        with ``network_head`` and ``image_encoder`` attributes).
    lr_ogd : float, optional
        Learning rate for the persistent supervised OGD optimizer, by
        default ``1e-4``.
    lr_tta : float, optional
        Learning rate for the ephemeral unsupervised TTA optimizer, by
        default ``1e-4``.
    tta_method : {'memo', 'note', 'none'}, optional
        Unsupervised TTA variant, by default ``'memo'``.
        - ``'memo'`` — per-sample augmented-view variance minimisation.
        - ``'note'`` — buffer-based variance minimisation.
        - ``'none'`` — no TTA; plain eval inference on ``base_model``.
    ogd_scope : {'head_only', 'head_and_encoder'}, optional
        Parameters permanently updated by OGD.
        - ``'head_only'`` — only the regression head.
        - ``'head_and_encoder'`` — head + the last child of ``image_encoder``.
        By default ``'head_only'``.
    ogd_loss : {'berhu', 'huber', 'mse', 'mae'}, optional
        Supervised loss function used during Phase 1 OGD, by default
        ``'mse'``.
    ogd_momentum : float, optional
        SGD momentum for the OGD optimizer, by default ``0.9``.
    ogd_weight_decay : float, optional
        Weight decay for the OGD optimizer, by default ``1e-4``.
    ogd_ramp_penalty : float, optional
        Multiplier for dynamic OGD loss scaling based on true irradiance
        delta. (0.0 = disabled), by default ``0.0``.
    tta_weight_decay : float, optional
        Weight decay for the ephemeral TTA AdamW optimizer, by default
        ``0.0``.
    n_augmentations : int, optional
        (MEMO) Number of augmented views generated per time step, by
        default ``8``.
    niter_tta : int, optional
        Number of TTA gradient steps applied to the deep copy per time
        step, by default ``1``.
    note_buffer_size : int, optional
        (NOTE) Maximum capacity of the persistent sample buffer, by
        default ``64``.
    """

    def __init__(
        self,
        base_model: nn.Module,
        lr_ogd: float = 1e-4,
        lr_tta: float = 1e-4,
        tta_method: Literal["memo", "note", "none"] = "memo",
        ogd_scope: Literal["head_only", "head_and_encoder"] = "head_only",
        ogd_loss: Literal["berhu", "huber", "mse", "mae"] = "mse",
        ogd_momentum: float = 0.9,
        ogd_weight_decay: float = 1e-4,
        ogd_ramp_penalty: float = 0.0,
        tta_weight_decay: float = 0.0,
        n_augmentations: int = 8,
        niter_tta: int = 1,
        note_buffer_size: int = 64,
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.lr_ogd = lr_ogd
        self.lr_tta = lr_tta
        self.tta_method = tta_method
        self.ogd_scope = ogd_scope
        self.ogd_momentum = ogd_momentum
        self.ogd_weight_decay = ogd_weight_decay
        self.ogd_ramp_penalty = ogd_ramp_penalty
        self.tta_weight_decay = tta_weight_decay
        self.n_augmentations = n_augmentations
        self.niter_tta = niter_tta
        self.note_buffer_size = note_buffer_size

        # ── Supervised OGD loss ──────────────────────────────────────────────
        if ogd_loss == "berhu":
            self._ogd_loss_fn: nn.Module = BerHuLoss(threshold=30)
        elif ogd_loss == "huber":
            self._ogd_loss_fn = nn.HuberLoss()
        elif ogd_loss == "mse":
            self._ogd_loss_fn = nn.MSELoss()
        elif ogd_loss == "mae":
            self._ogd_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported ogd_loss: {ogd_loss!r}")

        # ── NOTE persistent sample buffer ───────────────────────────────────
        # Stores (image [C,H,W], history [hist_len]) CPU tensor pairs.
        # Accumulates across steps (and optionally days) to provide a richer
        # batch for normalisation-layer adaptation.
        self._note_buffer: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=note_buffer_size)

        # ── Snapshot for resets ──────────────────────────────────────────────
        # Saved before the OGD optimizer is built so that reset() can
        # restore the original weights.
        self._initial_model_state = deepcopy(base_model.state_dict())

        # ── Persistent OGD optimizer (updates base_model in-place) ──────────
        # Built once; the same parameter objects are referenced throughout.
        self._ogd_optimizer: torch.optim.Optimizer = self._make_ogd_optimizer()

    # =========================================================================
    # Public interface
    # =========================================================================

    def reset(self) -> None:
        """Restore base_model to its initial checkpoint state.

        Also clears the NOTE sample buffer and recreates the OGD optimizer
        with a fresh state.
        """
        self.base_model.load_state_dict(self._initial_model_state, strict=True)
        self._note_buffer.clear()
        self._ogd_optimizer = self._make_ogd_optimizer()

    def update(
        self,
        image: torch.Tensor,  # [C, H, W]
        history: torch.Tensor,  # [hist_len]
        gt: torch.Tensor,  # [1, 1]
    ) -> None:
        """Phase B: one supervised OGD gradient step on base_model.

        Parameters
        ----------
        image : torch.Tensor, shape ``[C, H, W]``
            Sky image from the lagged observation.
        history : torch.Tensor, shape ``[hist_len]``
            Irradiance history vector from the lagged observation.
        gt : torch.Tensor, shape ``[1, 1]``
            Ground-truth label that is now causally observable.
        """
        self._ogd_step(image=image, history=history, gt=gt)

    def predict(
        self,
        image: torch.Tensor,  # [1, C, H, W]
        history: torch.Tensor,  # [1, hist_len]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Phases C+D: TTA on an ephemeral deepcopy, then final inference.

        Parameters
        ----------
        image : torch.Tensor, shape ``[1, C, H, W]``
            Current sky image (batch_size = 1 for online evaluation).
        history : torch.Tensor, shape ``[1, hist_len]``
            Current irradiance history vector.

        Returns
        -------
        mean : torch.Tensor, shape ``[1, 1]``
            Final calibrated forecast.
        variance : torch.Tensor or None, shape ``[1, 1]``
            Predictive variance if the model is probabilistic, else ``None``.
        """
        device = image.device
        img = image if image.dim() == 4 else image.unsqueeze(0)  # [1, C, H, W]
        hist = history if history.dim() == 2 else history.unsqueeze(0)  # [1, hist_len]

        if self.tta_method == "none":
            self.base_model.eval()
            with torch.no_grad():
                output = self.base_model(img.to(device), hist.to(device))
            mean = output[0] if isinstance(output, tuple) else output
            variance = output[1] if isinstance(output, tuple) else None
            return mean, variance

        # Phase C: Create an ephemeral deep copy and adapt its norm-layer affines.
        tta_model = deepcopy(self.base_model).to(device)
        tta_optimizer = self._make_tta_optimizer(tta_model)

        if self.tta_method == "memo":
            self._memo_tta_step(tta_model, tta_optimizer, img[0], hist, device)
        else:
            # NOTE: update the persistent buffer first, then adapt on the batch.
            self._note_buffer.append((img[0].detach().cpu(), hist[0].detach().cpu()))
            self._note_tta_step(tta_model, tta_optimizer, device)

        # Phase D: Final inference with the TTA-calibrated copy.
        tta_model.eval()
        with torch.no_grad():
            output = tta_model(img.to(device), hist.to(device))

        del tta_model, tta_optimizer

        if isinstance(output, tuple):
            mean, variance = output
        else:
            mean, variance = output, None

        return mean, variance

    # =========================================================================
    # Phase implementations
    # =========================================================================

    @torch.enable_grad()
    def _ogd_step(
        self,
        image: torch.Tensor,  # [C, H, W]
        history: torch.Tensor,  # [hist_len]
        gt: torch.Tensor,  # [1, 1]
    ) -> None:
        """Phase 1 — one supervised gradient step on base_model with Ramp Weighting.

        Configures the gradient flow so only OGD-scoped parameters receive
        gradients; all other parameters stay frozen.  Restores
        ``requires_grad=False`` on all params after stepping to prevent
        accidental gradient accumulation during the subsequent deepcopy.
        """
        self._set_ogd_mode()

        # Restore batch dimension (popped items are stored as [C,H,W] / [hist])
        imgs_batch = image.unsqueeze(0)  # [1, C, H, W]
        hist_batch = history.unsqueeze(0)  # [1, hist_len]

        self._ogd_optimizer.zero_grad()
        output = self.base_model(imgs_batch, hist_batch)
        pred = output[0] if isinstance(output, tuple) else output  # [1, 1]

        # 1. Base loss calculation
        loss = self._ogd_loss_fn(pred, gt.view_as(pred))

        # 2a. Ramp-Weighted Penalty (Adaptive loss scaling)
        if self.ogd_ramp_penalty > 0.0:
            # history[-1] is the most recent known irradiance (time t)
            # gt is the actual arrived irradiance (time t+H)
            current_irr = history[-1]
            gt_val = gt.view(-1)[0]

            # How much did the weather actually change in the real world?
            actual_delta = torch.abs(gt_val - current_irr)

            # Scale the loss dynamically
            ramp_weight = 1.0 + (actual_delta * self.ogd_ramp_penalty)

            # Apply the dynamic weight (detach to prevent graph issues)
            loss = loss * ramp_weight.detach()

        # 2b. Anti-smoothing volatility penalty
        current_irr = history[-1]
        scale_factor = 1000.0
        true_delta = (gt.view(-1)[0] - current_irr) * scale_factor
        pred_delta = (pred.view(-1)[0] - current_irr) * scale_factor

        slope_error = torch.abs(pred_delta - true_delta)

        history_phys = history * scale_factor
        hist_var = torch.var(history_phys, unbiased=False)

        volatility_multiplier = 0.09
        dynamic_slope_weight = 1.0 + (hist_var.detach() * volatility_multiplier)
        slope_loss = (slope_error * dynamic_slope_weight) / scale_factor

        loss = loss + slope_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
        self._ogd_optimizer.step()

        # Freeze all params again so the subsequent deepcopy does not carry
        # an active computation graph into the TTA copy.
        for param in self.base_model.parameters():
            param.requires_grad_(False)

    @torch.enable_grad()
    def _memo_tta_step(
        self,
        tta_model: nn.Module,
        tta_optimizer: torch.optim.Optimizer,
        img_single: torch.Tensor,  # [C, H, W]
        hist: torch.Tensor,  # [1, hist_len]
        device: torch.device,
    ) -> None:
        """Phase 2 (MEMO) — variance minimisation on N augmented views.

        Adapts the norm-layer affine parameters of ``tta_model`` (a deepcopy
        of ``base_model``) using the standard MEMO-Reg objective:

            L_TTA = Var({ŷ_i : i = 1..N})

        Generation of augmented views mirrors the training-time pipeline
        (random flip + affine ± 10°, ± 10% translate, ± 10% scale).
        """
        self._set_tta_mode(tta_model)

        for _ in range(self.niter_tta):
            # Stack N independently-augmented views from the single image.
            aug_imgs = torch.stack(
                [_random_augment(img_single) for _ in range(self.n_augmentations)],
                dim=0,
            ).to(device)  # [N, C, H, W]

            # Repeat the irradiance history for each augmented view.
            aug_hist = hist.expand(self.n_augmentations, -1).to(device)  # [N, hist_len]

            tta_optimizer.zero_grad()
            output = tta_model(aug_imgs, aug_hist)
            preds = output[0] if isinstance(output, tuple) else output  # [N, 1]
            loss = prediction_variance_loss(preds)
            loss.backward()
            tta_optimizer.step()

    @torch.enable_grad()
    def _note_tta_step(
        self,
        tta_model: nn.Module,
        tta_optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        """Phase 2 (NOTE) — variance minimisation on the persistent sample buffer.

        Adapts norm-layer affine parameters of ``tta_model`` using the full
        accumulated buffer as a batch.  The persistent buffer provides richer
        BN batch statistics than a single-sample augmentation approach,
        especially useful for non-i.i.d. temporal streams.

        Adaptation is skipped if the buffer holds fewer than
        :data:`_MIN_NOTE_BUFFER` samples, as BN running statistics are
        unreliable on very small batches.
        """
        if len(self._note_buffer) < _MIN_NOTE_BUFFER:
            # Too few samples for stable batch normalisation; skip this step.
            return

        self._set_tta_mode(tta_model)

        # Collate the buffer into a single tensor batch.
        buf_imgs = torch.stack([entry[0] for entry in self._note_buffer], dim=0).to(device)  # [B, C, H, W]
        buf_hist = torch.stack([entry[1] for entry in self._note_buffer], dim=0).to(device)  # [B, hist_len]

        for _ in range(self.niter_tta):
            tta_optimizer.zero_grad()
            output = tta_model(buf_imgs, buf_hist)
            preds = output[0] if isinstance(output, tuple) else output  # [B, 1]
            loss = prediction_variance_loss(preds)
            loss.backward()
            tta_optimizer.step()

    # =========================================================================
    # Mode helpers
    # =========================================================================

    def _set_ogd_mode(self) -> None:
        """Configure gradient flow on base_model for Phase 1 OGD.

        Freezes all parameters, then selectively unfreezes the regression head
        and (if ``ogd_scope='head_and_encoder'``) the deepest child of
        ``image_encoder``.  Puts ``base_model`` into train mode so BN layers
        update their running statistics during the supervised forward pass.
        """
        # Freeze all parameters to establish a clean baseline.
        for param in self.base_model.parameters():
            param.requires_grad_(False)

        # Always include the regression head.
        for param in self.base_model.network_head.parameters():
            param.requires_grad_(True)

        # Optionally include the deepest encoder block (last child of encoder).
        if self.ogd_scope == "head_and_encoder":
            encoder_children = list(self.base_model.image_encoder.children())
            if encoder_children:
                for i in range(-2, 0):  # last two children ([-1] is the deepest block, [-2] is its immediate predecessor)
                    for param in encoder_children[i].parameters():
                        param.requires_grad_(True)

        # Keep BN layers in eval mode so running statistics are not corrupted
        # by single-sample batches.  The head (Linear layers) has no BN, so
        # eval mode has no negative effect on OGD gradient quality.
        self.base_model.eval()

    @staticmethod
    def _set_tta_mode(model: nn.Module) -> None:
        """Configure gradient flow on a (deep-copied) model for Phase 2 TTA.

        Freezes all parameters, then unfreezes only the affine parameters
        (weight γ and bias β) of every BatchNorm1d / BatchNorm2d / LayerNorm
        module.  Puts the model in ``.eval()`` globally, then overrides each
        norm module back to ``.train()`` so that BN layers compute
        per-batch statistics from the augmented views rather than using stale
        running statistics.
        """
        # Freeze everything in the copy.
        for param in model.parameters():
            param.requires_grad_(False)

        # Selectively unfreeze affine params of normalisation layers.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.LayerNorm):
                affine = m.affine if hasattr(m, "affine") else (m.weight is not None)
                if affine:
                    if m.weight is not None:
                        m.weight.requires_grad_(True)
                    if m.bias is not None:
                        m.bias.requires_grad_(True)

        # Global eval (disables dropout etc.) then re-enable norm train mode.
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.LayerNorm):
                m.train()

    # =========================================================================
    # Optimizer factories
    # =========================================================================

    def _make_ogd_optimizer(self) -> torch.optim.Optimizer:
        """Build the persistent SGD optimizer for OGD-scoped base_model params.

        The optimizer references the *same* parameter tensor objects that live
        in ``base_model``, so it remains valid for the lifetime of the adapter
        without requiring a rebuild after each step.
        """
        params = list(self.base_model.network_head.parameters())
        if self.ogd_scope == "head_and_encoder":
            encoder_children = list(self.base_model.image_encoder.children())
            if encoder_children:
                for i in range(-2, 0):
                    params.extend(encoder_children[i].parameters())
        return torch.optim.SGD(
            params,
            lr=self.lr_ogd,
            momentum=self.ogd_momentum,
            weight_decay=self.ogd_weight_decay,
        )

    def _make_tta_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build a fresh AdamW optimizer for norm-layer affine params of model.

        Always constructs a new optimizer pointing at the affine parameters of
        the provided (deep-copied) model.  AdamW is preferred over plain Adam
        for norm-layer-only adaptation: the decoupled weight decay provides a
        mild regularising pull toward the pre-TTA affine values.
        """
        params: list[nn.Parameter] = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.LayerNorm):
                affine = m.affine if hasattr(m, "affine") else (m.weight is not None)
                if affine:
                    if m.weight is not None:
                        params.append(m.weight)
                    if m.bias is not None:
                        params.append(m.bias)

        if not params:
            # No normalisation layers found — fall back to the regression head
            # so the TTA step is always well-defined.
            params = list(model.network_head.parameters())

        return torch.optim.AdamW(params, lr=self.lr_tta, weight_decay=self.tta_weight_decay)

    # =========================================================================
    # Repr
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"DualPhaseAdapter("
            f"tta_method={self.tta_method!r}, "
            f"ogd_scope={self.ogd_scope!r}, "
            f"lr_ogd={self.lr_ogd}, lr_tta={self.lr_tta}, "
            f"ogd_ramp_penalty={self.ogd_ramp_penalty}, "
            f"n_augmentations={self.n_augmentations}, "
            f"niter_tta={self.niter_tta})"
        )
