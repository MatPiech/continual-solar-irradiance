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
"""

from collections import deque
from copy import deepcopy
from typing import Literal

import torch
import torch.nn as nn

from cirr.training.ogd import make_regression_loss, ogd_step
from cirr.tta.common import prediction_variance_loss
from cirr.tta.memo_reg import _random_augment

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
    """Dual-Phase Online Adapter: persistent OGD + ephemeral TTA."""

    def __init__(
        self,
        base_model: nn.Module,
        lr_ogd: float = 1e-4,
        lr_tta: float = 1e-4,
        tta_method: Literal["memo", "note", "aug_ensemble", "none"] = "none",
        ogd_scope: Literal["head_only", "head_and_encoder"] = "head_only",
        ogd_loss: Literal["berhu", "huber", "mse"] = "berhu",
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
        if ogd_loss == "mae":
            self._ogd_loss_fn: nn.Module = nn.L1Loss()
        else:
            self._ogd_loss_fn = make_regression_loss(ogd_loss)

        # ── NOTE persistent sample buffer ───────────────────────────────────
        self._note_buffer: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=note_buffer_size)

        # ── Snapshot for resets ──────────────────────────────────────────────
        self._initial_model_state = deepcopy(base_model.state_dict())

        # ── Persistent OGD optimizer (updates base_model in-place) ──────────
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

    def _ogd_step(
        self,
        image: torch.Tensor,
        history: torch.Tensor,
        gt: torch.Tensor,
    ) -> None:
        device = str(next(self.base_model.parameters()).device)
        ogd_step(
            self.base_model,
            self._ogd_optimizer,
            self.ogd_scope,
            self._ogd_loss_fn,
            image,
            history,
            gt,
            device,
            slope_penalty=True,
            max_grad_norm=10.0,
            encoder_unfreeze="full",
        )

    def predict(
        self,
        image: torch.Tensor,  # [1, C, H, W]
        history: torch.Tensor,  # [1, hist_len]
        **model_kwargs,
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
                output = self.base_model(img.to(device), hist.to(device), **model_kwargs)
            mean = output[0] if isinstance(output, tuple) else output
            variance = output[1] if isinstance(output, tuple) else None
            return mean, variance

        if self.tta_method == "aug_ensemble":
            # 1. Pure Inference: No deepcopy, no gradients, no optimizers.
            self.base_model.eval()

            # 2. Generate augmented views
            aug_imgs = torch.stack(
                [_random_augment(img[0]) for _ in range(self.n_augmentations)],
                dim=0,
            ).to(device)
            aug_hist = hist.expand(self.n_augmentations, -1).to(device)

            # 3. Predict on all views and average the result
            with torch.no_grad():
                output = self.base_model(aug_imgs, aug_hist)
                preds = output[0] if isinstance(output, tuple) else output

                # The mean of the augmentations acts as a spatial low-pass filter
                # smoothing out sensor noise without destroying the BN weights.
                ensemble_mean = preds.mean(dim=0, keepdim=True)

            return ensemble_mean, None

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
            output = tta_model(img.to(device), hist.to(device), **model_kwargs)

        del tta_model, tta_optimizer

        if isinstance(output, tuple):
            mean, variance = output
        else:
            mean, variance = output, None

        return mean, variance

    @torch.no_grad()
    def predict_base(
        self,
        image: torch.Tensor,  # [1, C, H, W]
        history: torch.Tensor,  # [1, hist_len]
        **model_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run base_model inference without any TTA adaptation.

        Useful for lightweight online signals (e.g. delayed gating/calibration)
        when the final served prediction is produced by another branch.
        """
        device = image.device
        img = image if image.dim() == 4 else image.unsqueeze(0)
        hist = history if history.dim() == 2 else history.unsqueeze(0)

        self.base_model.eval()
        output = self.base_model(img.to(device), hist.to(device), **model_kwargs)
        mean = output[0] if isinstance(output, tuple) else output
        variance = output[1] if isinstance(output, tuple) else None
        return mean, variance

    @torch.no_grad()
    def predict_base_with_features(
        self,
        image: torch.Tensor,
        history: torch.Tensor,
        **model_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run base_model inference without TTA and return encoder features."""
        device = image.device
        img = image if image.dim() == 4 else image.unsqueeze(0)
        hist = history if history.dim() == 2 else history.unsqueeze(0)

        self.base_model.eval()
        if hasattr(self.base_model, "forward_with_encoder_features"):
            mean, variance, features = self.base_model.forward_with_encoder_features(
                img.to(device),
                hist.to(device),
                **model_kwargs,
            )
            return mean, variance, features

        output = self.base_model(img.to(device), hist.to(device), **model_kwargs)
        mean = output[0] if isinstance(output, tuple) else output
        variance = output[1] if isinstance(output, tuple) else None
        return mean, variance, None

    def predict_with_features(
        self,
        image: torch.Tensor,
        history: torch.Tensor,
        **model_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run prediction and additionally return encoder features when available."""
        device = image.device
        img = image if image.dim() == 4 else image.unsqueeze(0)
        hist = history if history.dim() == 2 else history.unsqueeze(0)

        if self.tta_method == "none":
            self.base_model.eval()
            with torch.no_grad():
                if hasattr(self.base_model, "forward_with_encoder_features"):
                    mean, variance, features = self.base_model.forward_with_encoder_features(
                        img.to(device),
                        hist.to(device),
                        **model_kwargs,
                    )
                else:
                    output = self.base_model(img.to(device), hist.to(device), **model_kwargs)
                    mean = output[0] if isinstance(output, tuple) else output
                    variance = output[1] if isinstance(output, tuple) else None
                    features = None
            return mean, variance, features

        if self.tta_method == "aug_ensemble":
            self.base_model.eval()

            aug_imgs = torch.stack(
                [_random_augment(img[0]) for _ in range(self.n_augmentations)],
                dim=0,
            ).to(device)
            aug_hist = hist.expand(self.n_augmentations, -1).to(device)

            with torch.no_grad():
                output = self.base_model(aug_imgs, aug_hist)
                preds = output[0] if isinstance(output, tuple) else output
                ensemble_mean = preds.mean(dim=0, keepdim=True)

                if hasattr(self.base_model, "forward_with_encoder_features"):
                    _, _, features = self.base_model.forward_with_encoder_features(
                        img.to(device),
                        hist.to(device),
                        **model_kwargs,
                    )
                else:
                    features = None

            return ensemble_mean, None, features

        tta_model = deepcopy(self.base_model).to(device)
        tta_optimizer = self._make_tta_optimizer(tta_model)

        if self.tta_method == "memo":
            self._memo_tta_step(tta_model, tta_optimizer, img[0], hist, device)
        else:
            self._note_buffer.append((img[0].detach().cpu(), hist[0].detach().cpu()))
            self._note_tta_step(tta_model, tta_optimizer, device)

        tta_model.eval()
        with torch.no_grad():
            if hasattr(tta_model, "forward_with_encoder_features"):
                mean, variance, features = tta_model.forward_with_encoder_features(
                    img.to(device),
                    hist.to(device),
                    **model_kwargs,
                )
            else:
                output = tta_model(img.to(device), hist.to(device), **model_kwargs)
                mean = output[0] if isinstance(output, tuple) else output
                variance = output[1] if isinstance(output, tuple) else None
                features = None

        del tta_model, tta_optimizer
        return mean, variance, features

    # =========================================================================
    # Phase implementations
    # =========================================================================

    @torch.enable_grad()
    def _memo_tta_step(
        self,
        tta_model: nn.Module,
        tta_optimizer: torch.optim.Optimizer,
        img_single: torch.Tensor,
        hist: torch.Tensor,
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
            aug_imgs = torch.stack(
                [_random_augment(img_single) for _ in range(self.n_augmentations)],
                dim=0,
            ).to(device)

            aug_hist = hist.expand(self.n_augmentations, -1).to(device)

            tta_optimizer.zero_grad()
            output = tta_model(aug_imgs, aug_hist)
            preds = output[0] if isinstance(output, tuple) else output
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
        """Phase 2 (NOTE) — variance minimisation on buffered samples.

        Adapts the norm-layer affine parameters of ``tta_model`` (a deepcopy
        of ``base_model``) using the standard NOTE-Reg objective:

            L_TTA = Var({ŷ_i : i = 1..N})

        where ``N`` is the number of samples in the buffer.
        """
        if len(self._note_buffer) < _MIN_NOTE_BUFFER:
            return

        self._set_tta_mode(tta_model)

        buf_imgs = torch.stack([entry[0] for entry in self._note_buffer], dim=0).to(device)
        buf_hist = torch.stack([entry[1] for entry in self._note_buffer], dim=0).to(device)

        for _ in range(self.niter_tta):
            tta_optimizer.zero_grad()
            output = tta_model(buf_imgs, buf_hist)
            preds = output[0] if isinstance(output, tuple) else output
            loss = prediction_variance_loss(preds)
            loss.backward()
            tta_optimizer.step()

    # =========================================================================
    # Mode helpers
    # =========================================================================

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

        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.LayerNorm):
                affine = m.affine if hasattr(m, "affine") else (m.weight is not None)
                if affine:
                    if m.weight is not None:
                        m.weight.requires_grad_(True)
                    if m.bias is not None:
                        m.bias.requires_grad_(True)

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
            params.extend(self.base_model.image_encoder.parameters())
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
            params = list(model.network_head.parameters())

        return torch.optim.AdamW(params, lr=self.lr_tta, weight_decay=self.tta_weight_decay)

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
