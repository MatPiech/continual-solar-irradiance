"""NOTE-Reg: Non-i.i.d. Online Test-time adaptation for Regression.

Adapts NOTE (Gong et al., NeurIPS 2022) from classification to regression.
The original NOTE handles non-i.i.d. test streams by maintaining a FIFO memory
buffer of recent samples and adapting BatchNorm statistics using entropy
minimisation on that buffer.  For regression — where entropy is undefined —
NOTE-Reg replaces the entropy objective with *prediction variance minimisation*:

    Loss = (1/N) * Σ_i (ŷ_i − ȳ)²,   ȳ = (1/N) * Σ_i ŷ_i

applied to the buffered batch every ``update_every_x`` samples.

Key differences from MEMO-Reg
------------------------------
* **Not episodic by default**: weights accumulate adaptation over the test
  stream rather than being reset per sample.
* **Memory buffer**: a FIFO deque of recent ``(image, irradiance_history)``
  pairs drives each adaptation step (batch-level, not single-sample-level).
* **Update frequency**: adaptation is triggered after every
  ``update_every_x`` samples, not on every single sample.
* **BN behaviour**: two modes controlled by ``use_learned_stats``:
  - ``True`` — BN layers track running statistics with given ``bn_momentum``;
    predictions use the accumulated running mean/var.
  - ``False`` — BN layers use per-batch statistics at both train and test
    time (running_mean / running_var are disabled).  Matches the
    "batch-free inference" branch in the original NOTE.

Procedure (online, one sample at a time)
------------------------------------------
1. Receive sky image *I_t* and irradiance history *h_t*.
2. Append (*I_t*, *h_t*) to the FIFO buffer (evicting oldest if full).
3. If ``t mod update_every_x == 0``: run *K* gradient steps on the full buffer
   using prediction variance loss.
4. Return the prediction for *I_t* using the current (adapted) model.

References
----------
.. [1] Gong, T., et al. (2022).
       "NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation."
       NeurIPS 2022. https://arxiv.org/abs/2208.05117
"""

from collections import deque
from copy import deepcopy
from typing import Literal

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Variance-minimisation loss  (shared objective with MEMO-Reg)
# ---------------------------------------------------------------------------


def prediction_variance_loss(predictions: torch.Tensor) -> torch.Tensor:
    """Mean prediction variance across a batch.

    Parameters
    ----------
    predictions : torch.Tensor
        Shape ``[N, 1]`` or ``[N]``.

    Returns
    -------
    torch.Tensor
        Scalar variance loss.
    """
    preds = predictions.squeeze(-1)  # [N]
    mean_pred = preds.mean()
    return ((preds - mean_pred) ** 2).mean()


# ---------------------------------------------------------------------------
# NOTE-Reg
# ---------------------------------------------------------------------------


class NOTEReg(nn.Module):
    """NOTE for regression: memory-buffered online BN adaptation.

    At inference time, arriving sky images are collected into a FIFO deque.
    Every ``update_every_x`` samples the model's parameters (BN-only by
    default) are updated via gradient descent to minimise prediction variance
    on the buffered batch.  Each sample's final forecast is then produced by
    the current (adapted) model state.

    Parameters
    ----------
    model : nn.Module
        The ``Forecaster`` (or compatible) model to adapt.
    lr : float, optional
        Adaptation learning rate, by default 1e-4.
    weight_decay : float, optional
        Optimizer weight decay, by default 0.0.
    update_every_x : int, optional
        Number of samples between adaptation steps (also the FIFO buffer
        capacity), by default 64.
    niter : int, optional
        Number of gradient steps per adaptation trigger, by default 1.
    adapt_mode : {'bn_only', 'all'}, optional
        Which parameters to update.
        - ``'bn_only'``: only BatchNorm affine params (γ, β) — the original
          NOTE default and recommended for distribution-shift robustness.
        - ``'all'``: all model parameters.
        By default ``'bn_only'``.
    use_learned_stats : bool, optional
        Controls BN statistics mode during adaptation:
        - ``True`` — BN tracks running stats (momentum = ``bn_momentum``);
          inference uses accumulated mean / var.
        - ``False`` — BN uses per-batch statistics, running mean/var are
          disabled; every forward pass uses the current-batch statistics.
        By default ``True``.
    bn_momentum : float, optional
        Momentum for running statistics when ``use_learned_stats=True``,
        by default 0.1.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        update_every_x: int = 64,
        niter: int = 1,
        adapt_mode: Literal["bn_only", "all"] = "bn_only",
        use_learned_stats: bool = True,
        bn_momentum: float = 0.1,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.update_every_x = update_every_x
        self.niter = niter
        self.adapt_mode = adapt_mode
        self.use_learned_stats = use_learned_stats
        self.bn_momentum = bn_momentum

        # Save initial weights for full reset
        self._initial_model_state = deepcopy(model.state_dict())

        # FIFO buffer — stores CPU tensors to avoid holding GPU memory
        self._buffer_images: deque[torch.Tensor] = deque(maxlen=update_every_x)
        self._buffer_histories: deque[torch.Tensor] = deque(maxlen=update_every_x)
        self._sample_count: int = 0

        # Validate and configure
        self.check_model()
        self.configure_model()

        # Optimizer (created after configure_model sets requires_grad)
        self.optimizer = torch.optim.Adam(_get_params(model, adapt_mode), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        irradiance_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Online test-time forward: buffer, (optionally) adapt, predict.

        Processes each image in the batch independently in the order it
        appears (reflecting the online, one-at-a-time nature of NOTE).

        Parameters
        ----------
        images : torch.Tensor
            ``[batch_size, c, h, w]`` — typically batch_size=1 for online TTA.
        irradiance_history : torch.Tensor
            ``[batch_size, history_len]``.

        Returns
        -------
        means : torch.Tensor
            Predicted mean irradiance values, shape ``[batch_size, 1]``.
        variances : torch.Tensor or None
            Predicted aleatoric variances, or ``None``.
        """
        device = images.device
        batch_size = images.size(0)
        means_list: list[torch.Tensor] = []
        vars_list: list[torch.Tensor | None] = []

        for i in range(batch_size):
            img_i = images[i]  # [c, h, w]
            hist_i = irradiance_history[i]  # [history_len]

            # ---- 1. Buffer ----
            self._buffer_images.append(img_i.detach().cpu())
            self._buffer_histories.append(hist_i.detach().cpu())
            self._sample_count += 1

            # ---- 2. Adapt (triggered every update_every_x samples) ----
            if self._sample_count % self.update_every_x == 0:
                self._adapt_from_buffer(device)

            # ---- 3. Predict ----
            mean_i, var_i = self._predict_single(img_i, hist_i, device)
            means_list.append(mean_i)
            vars_list.append(var_i)

        means = torch.cat(means_list, dim=0)  # [batch_size, 1]
        variances = torch.cat(vars_list, dim=0) if vars_list[0] is not None else None
        return means, variances

    def reset(self) -> None:
        """Reset model, optimizer, and buffer to initial state."""
        self.model.load_state_dict(self._initial_model_state, strict=True)
        self._buffer_images.clear()
        self._buffer_histories.clear()
        self._sample_count = 0
        params = _get_params(self.model, self.adapt_mode)
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        self.configure_model()

    def configure_model(self) -> nn.Module:
        """Configure trainability and BN statistics mode."""
        if self.adapt_mode == "bn_only":
            # Freeze everything …
            for param in self.model.parameters():
                param.requires_grad_(False)
            # … then selectively enable BN affine params and set BN mode
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d):
                    if self.use_learned_stats:
                        m.track_running_stats = True
                        m.momentum = self.bn_momentum
                    else:
                        m.track_running_stats = False
                        m.running_mean = None
                        m.running_var = None
                    if m.affine:
                        m.weight.requires_grad_(True)
                        m.bias.requires_grad_(True)
            self.model.eval()
            # Keep BN layers in train mode so they update statistics
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d):
                    m.train()
        else:
            # 'all': full model trains
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad_(True)
        return self.model

    def check_model(self) -> None:
        """Validate model compatibility for ``bn_only`` mode."""
        if self.adapt_mode == "bn_only":
            has_bn = any(isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d) for m in self.model.modules())
            if not has_bn:
                raise AssertionError(
                    "NOTEReg with adapt_mode='bn_only' requires BatchNorm layers. "
                    "Use adapt_mode='all' for models without BN."
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def _adapt_from_buffer(self, device: torch.device) -> None:
        """Run adaptation steps on the current FIFO buffer contents."""
        imgs = torch.stack(list(self._buffer_images)).to(device)  # [B, c, h, w]
        hists = torch.stack(list(self._buffer_histories)).to(device)  # [B, history_len]

        for _ in range(self.niter):
            self.configure_model()
            self.optimizer.zero_grad()
            output = self.model(imgs, hists)
            preds = output[0] if isinstance(output, tuple) else output  # [B, 1]
            loss = prediction_variance_loss(preds)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def _predict_single(
        self,
        img: torch.Tensor,
        hist: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict on a single (original) image using the current model state."""
        self.model.eval()
        x = img.unsqueeze(0).to(device)  # [1, c, h, w]
        h = hist.unsqueeze(0).to(device)  # [1, history_len]
        output = self.model(x, h)

        if isinstance(output, tuple):
            mean, variance = output  # [1, 1], [1, 1] or similar
        else:
            mean, variance = output, None

        return mean, variance

    def __repr__(self) -> str:
        return (
            f"NOTEReg(update_every_x={self.update_every_x}, "
            f"niter={self.niter}, lr={self.lr}, adapt_mode={self.adapt_mode!r}, "
            f"use_learned_stats={self.use_learned_stats}, "
            f"bn_momentum={self.bn_momentum}, "
            f"buffer_size={len(self._buffer_images)}/{self.update_every_x})"
        )


# ---------------------------------------------------------------------------
# Helper: select parameters for the optimizer
# ---------------------------------------------------------------------------


def _get_params(model: nn.Module, adapt_mode: str) -> list[nn.Parameter]:
    """Return the list of parameters to optimise."""
    if adapt_mode == "bn_only":
        params: list[nn.Parameter] = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d) and m.affine:
                params.extend([m.weight, m.bias])
        return params
    else:
        return list(model.parameters())
