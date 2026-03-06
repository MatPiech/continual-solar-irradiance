"""Dual-Phase Online Adaptation for solar irradiance forecasting.

Combines two complementary adaptation signals at every time step:

Phase 1 — Lagged Supervised Online Gradient Descent (OGD)
----------------------------------------------------------
When the true irradiance y_t arrives (after the 15-minute prediction horizon),
the matching inputs (I_{t-15}, H_{t-15}) are retrieved from a FIFO cache and
used for one supervised gradient step with BerHu (or Huber) loss.

Update scope: the regression head, and optionally the last block of the image
encoder (``ogd_scope='head_and_encoder'``).

Phase 2 — Zero-Lag Unsupervised TTA (MEMO-Reg)
------------------------------------------------
Immediately after OGD the *current* image I_t is augmented N times.  The
variance of the resulting predictions drives one unsupervised gradient step:

    Loss_TTA = Var({ŷ_i}_{i=1..N})

Update scope: affine parameters (γ, β) of all BatchNorm / LayerNorm layers
only, leaving all conv / attention weights frozen.

Phase 3 — Final Inference + Cache
-----------------------------------
The original (unaugmented) inputs are forwarded through the fully-updated
model to yield the final forecast ŷ_{t+15}.  The current inputs are pushed
into the FIFO queue for use in 15 minutes' time.

Algorithm reference
--------------------
.. [1] Zhang et al. "MEMO: Test Time Robustness via Adaptation and
       Augmentation." NeurIPS 2022. https://arxiv.org/abs/2110.09506

Architecture assumptions
-------------------------
The wrapped model must expose:
* ``model.network_head``   — nn.Sequential regression head
* ``model.image_encoder``  — timm image encoder

These are satisfied by :class:`cirr.models.forecaster.Forecaster`.
"""

from collections import deque
from copy import deepcopy
from typing import Literal

import torch
import torch.nn as nn

from cirr.losses import MeanAdaptiveBerHuLoss
from cirr.tta.memo_reg import _random_augment, prediction_variance_loss

# ---------------------------------------------------------------------------
# Dual-Phase Adapter
# ---------------------------------------------------------------------------


class DualPhaseAdapter(nn.Module):
    """Dual-Phase Online Adaptation: Lagged OGD + Zero-Lag TTA.

    Parameters
    ----------
    model : nn.Module
        A :class:`~cirr.models.forecaster.Forecaster` instance (or any model
        with ``network_head`` and ``image_encoder`` attributes).
    cache_size : int, optional
        FIFO depth in time steps.  With 5-minute data intervals and a
        15-minute prediction horizon set this to ``3``, by default 3.
    ogd_lr : float, optional
        Learning rate for the supervised OGD optimizer, by default 1e-4.
    ogd_momentum : float, optional
        SGD momentum for OGD, by default 0.9.
    ogd_weight_decay : float, optional
        Weight decay for OGD, by default 1e-4.
    ogd_scope : {'head_only', 'head_and_encoder'}, optional
        Parameters updated by OGD.
        - ``'head_only'``: only the regression head.
        - ``'head_and_encoder'``: head + the last top-level child module of
          the image encoder (typically the deepest residual/attention block).
        By default ``'head_only'``.
    ogd_loss : {'berhu', 'huber'}, optional
        Supervised loss for OGD.  ``'berhu'`` uses
        :class:`~cirr.losses.MeanAdaptiveBerHuLoss`; ``'huber'`` uses
        ``torch.nn.HuberLoss``, by default ``'berhu'``.
    tta_lr : float, optional
        Learning rate for the unsupervised TTA optimizer, by default 1e-5.
    tta_weight_decay : float, optional
        Weight decay for TTA, by default 0.0.
    n_augmentations : int, optional
        Number of augmented views for the TTA variance loss, by default 8.
    niter_tta : int, optional
        Number of TTA gradient steps per time step, by default 1.
    """

    def __init__(
        self,
        model: nn.Module,
        cache_size: int = 3,
        ogd_lr: float = 1e-4,
        ogd_momentum: float = 0.9,
        ogd_weight_decay: float = 1e-4,
        ogd_scope: Literal["head_only", "head_and_encoder"] = "head_only",
        ogd_loss: Literal["berhu", "huber"] = "berhu",
        tta_lr: float = 1e-5,
        tta_weight_decay: float = 0.0,
        n_augmentations: int = 8,
        niter_tta: int = 1,
    ):
        super().__init__()

        self.model = model
        self.cache_size = cache_size
        self.ogd_lr = ogd_lr
        self.ogd_momentum = ogd_momentum
        self.ogd_weight_decay = ogd_weight_decay
        self.ogd_scope = ogd_scope
        self.tta_lr = tta_lr
        self.tta_weight_decay = tta_weight_decay
        self.n_augmentations = n_augmentations
        self.niter_tta = niter_tta

        # Supervised loss for OGD
        if ogd_loss == "berhu":
            self._ogd_loss_fn: nn.Module = MeanAdaptiveBerHuLoss()
        else:
            self._ogd_loss_fn = nn.HuberLoss()

        # FIFO input cache: stores CPU tensors (image, history)
        self._cache: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=cache_size)

        # Save initial weights for full reset
        self._initial_model_state = deepcopy(model.state_dict())

        # Build both optimizers (after saving initial state)
        self._ogd_optimizer = self._make_ogd_optimizer()
        self._tta_optimizer = self._make_tta_optimizer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        irradiance_history: torch.Tensor,
        ground_truth: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Online forward pass: adapt then predict.

        Call this once per arriving sample in chronological order.

        Parameters
        ----------
        images : torch.Tensor
            Current sky image batch ``[1, C, H, W]`` (batch_size=1 for online).
        irradiance_history : torch.Tensor
            Current irradiance history ``[1, history_len]``.
        ground_truth : torch.Tensor or None, optional
            True irradiance value ``[1, 1]`` for the *current* time step (i.e.
            the label that was predicted 15 minutes ago).  When provided and
            the cache is full, triggers Phase 1 OGD.

        Returns
        -------
        means : torch.Tensor, shape ``[1, 1]``
        variances : torch.Tensor or None, shape ``[1, 1]``
        """
        device = images.device

        # ── Phase 1: Lagged Supervised OGD ──────────────────────────────
        # Requires: (a) ground truth available, (b) cache has a lagged entry
        if ground_truth is not None and len(self._cache) == self._cache.maxlen:
            cached_img, cached_hist = self._cache[0]  # oldest = t - cache_size steps
            self._ogd_step(
                cached_img.to(device),
                cached_hist.to(device),
                ground_truth.to(device),
            )

        # ── Phase 2: Zero-Lag Unsupervised TTA ─────────────────────────
        img_single = images[0]  # [C, H, W]
        hist_single = irradiance_history  # [1, history_len]
        self._tta_step(img_single, hist_single, device)

        # ── Phase 3: Final Inference ────────────────────────────────────
        mean, variance = self._predict(images, irradiance_history, device)

        # ── Cache current inputs (for OGD 15 min from now) ─────────────
        self._cache.append((images[0].detach().cpu(), irradiance_history[0].detach().cpu()))

        return mean, variance

    def reset(self) -> None:
        """Restore model to initial checkpoint state and clear the input cache."""
        self.model.load_state_dict(self._initial_model_state, strict=True)
        self._cache.clear()
        self._ogd_optimizer = self._make_ogd_optimizer()
        self._tta_optimizer = self._make_tta_optimizer()

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def _ogd_step(
        self,
        cached_img: torch.Tensor,
        cached_hist: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> None:
        """Phase 1: one supervised gradient step with lagged inputs."""
        self._set_ogd_mode()

        images_batch = cached_img.unsqueeze(0)  # [1, C, H, W]
        hist_batch = cached_hist.unsqueeze(0)  # [1, history_len]

        self._ogd_optimizer.zero_grad()
        output = self.model(images_batch, hist_batch)
        pred = output[0] if isinstance(output, tuple) else output  # [1, 1]

        gt = ground_truth.view_as(pred)
        loss = self._ogd_loss_fn(pred, gt)
        loss.backward()
        self._ogd_optimizer.step()

    @torch.enable_grad()
    def _tta_step(
        self,
        img: torch.Tensor,
        hist: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Phase 2: one unsupervised variance-minimisation step."""
        self._set_tta_mode()

        for _ in range(self.niter_tta):
            aug_imgs = torch.stack([_random_augment(img) for _ in range(self.n_augmentations)], dim=0).to(
                device
            )  # [N, C, H, W]
            aug_hist = hist.expand(self.n_augmentations, -1)  # [N, history_len]

            self._tta_optimizer.zero_grad()
            output = self.model(aug_imgs, aug_hist)
            preds = output[0] if isinstance(output, tuple) else output  # [N, 1]
            loss = prediction_variance_loss(preds)
            loss.backward()
            self._tta_optimizer.step()

    @torch.no_grad()
    def _predict(
        self,
        images: torch.Tensor,
        irradiance_history: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Phase 3: final inference on the original unaugmented input."""
        self.model.eval()
        x = images.to(device)
        h = irradiance_history.to(device)
        output = self.model(x, h)
        if isinstance(output, tuple):
            mean, variance = output
        else:
            mean, variance = output, None
        return mean, variance

    # ------------------------------------------------------------------
    # Mode configuration helpers
    # ------------------------------------------------------------------

    def _set_ogd_mode(self) -> None:
        """Freeze everything; unfreeze only OGD-scope parameters."""
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Always adapt the regression head
        for param in self.model.network_head.parameters():
            param.requires_grad_(True)

        # Optionally adapt the deepest encoder block
        if self.ogd_scope == "head_and_encoder":
            encoder_children = list(self.model.image_encoder.children())
            if encoder_children:
                for param in encoder_children[-1].parameters():
                    param.requires_grad_(True)

        self.model.train()

    def _set_tta_mode(self) -> None:
        """Freeze everything; unfreeze only norm layer affine params (γ, β)."""
        for param in self.model.parameters():
            param.requires_grad_(False)

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.LayerNorm):
                if m.affine if hasattr(m, "affine") else (m.weight is not None):
                    if m.weight is not None:
                        m.weight.requires_grad_(True)
                    if m.bias is not None:
                        m.bias.requires_grad_(True)
                # Keep BN in train mode so it updates / uses batch statistics
                m.train()

        # All other sub-modules stay in eval mode
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.LayerNorm):
                m.train()

    # ------------------------------------------------------------------
    # Optimizer factories
    # ------------------------------------------------------------------

    def _make_ogd_optimizer(self) -> torch.optim.Optimizer:
        """Create a fresh SGD optimizer for OGD-scope parameters."""
        params = list(self.model.network_head.parameters())
        if self.ogd_scope == "head_and_encoder":
            encoder_children = list(self.model.image_encoder.children())
            if encoder_children:
                params.extend(encoder_children[-1].parameters())
        return torch.optim.SGD(
            params,
            lr=self.ogd_lr,
            momentum=self.ogd_momentum,
            weight_decay=self.ogd_weight_decay,
        )

    def _make_tta_optimizer(self) -> torch.optim.Optimizer:
        """Create a fresh Adam optimizer for norm-layer affine parameters."""
        params: list[nn.Parameter] = []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.LayerNorm):
                affine = m.affine if hasattr(m, "affine") else (m.weight is not None)
                if affine:
                    if m.weight is not None:
                        params.append(m.weight)
                    if m.bias is not None:
                        params.append(m.bias)
        if not params:
            # Fallback: if no norm layers exist, adapt all head params via TTA too
            params = list(self.model.network_head.parameters())
        return torch.optim.Adam(params, lr=self.tta_lr, weight_decay=self.tta_weight_decay)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DualPhaseAdapter("
            f"cache_size={self.cache_size}, "
            f"ogd_lr={self.ogd_lr}, ogd_scope={self.ogd_scope!r}, "
            f"tta_lr={self.tta_lr}, n_augmentations={self.n_augmentations}, "
            f"niter_tta={self.niter_tta})"
        )
