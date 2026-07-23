"""Sherman-Morrison RLS head adaptation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cirr.models.forecaster import Forecaster


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
        max_last_layer_update_norm: float = 0.01,
        max_first_layer_update_norm: float = 0.002,
        anchor_decay_last: float = 1e-4,
        anchor_decay_first: float = 5e-4,
        device: str = "cuda",
    ) -> None:
        self.lam = forgetting_factor
        self.device = device
        self.error_clip = float(error_clip)
        self.max_last_layer_update_norm = float(max_last_layer_update_norm)
        self.max_first_layer_update_norm = float(max_first_layer_update_norm)
        self.anchor_decay_last = float(anchor_decay_last)
        self.anchor_decay_first = float(anchor_decay_first)
        self._lambda_init = float(lambda_init)
        self._full_head_regularization_mult = float(full_head_regularization_mult)

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

        # Anchor tensors (checkpoint state) used for mild anti-drift pullback.
        self._last_weight_anchor = self._last_linear.weight.detach().clone()
        self._last_bias_anchor = self._last_linear.bias.detach().clone() if self._use_bias_last else None
        self._first_weight_anchor = self._first_linear.weight.detach().clone()
        self._first_bias_anchor = self._first_linear.bias.detach().clone() if self._use_bias_first else None

    # ------------------------------------------------------------------

    def _sm_update(self, p: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _clip_delta(self, delta: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Clip update tensor by global L2 norm."""
        norm = float(torch.linalg.norm(delta).item())
        if norm <= max_norm or norm < 1e-12:
            return delta
        return delta * (max_norm / norm)

    @torch.no_grad()
    def reset_covariance(self) -> None:
        """Reset only RLS covariance state, keeping adapted weights."""
        self._P2 = torch.eye(self._P2.shape[0], dtype=self._P2.dtype, device=self._P2.device) / self._lambda_init
        self._P1 = torch.eye(self._P1.shape[0], dtype=self._P1.dtype, device=self._P1.device) / (
            self._lambda_init * self._full_head_regularization_mult
        )

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
        img_feat = model.encode_fusion_features(imgs, hist)  # [1, num_features]
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

        self._P2, k2 = self._sm_update(self._P2, phi2_eff)

        # Keep a pre-update snapshot for hidden-layer credit propagation.
        w2_pre = self._last_linear.weight.detach().clone().to(self.device)

        if self._use_bias_last:
            delta_w2 = (k2[:-1] * error).unsqueeze(0)
            delta_w2 = self._clip_delta(delta_w2, self.max_last_layer_update_norm)
            self._last_linear.weight.add_(delta_w2.to(self._last_linear.weight.device))

            delta_b2 = (k2[-1] * error).view(1)
            delta_b2 = self._clip_delta(delta_b2, self.max_last_layer_update_norm)
            self._last_linear.bias.add_(delta_b2[0].to(self._last_linear.bias.device))
        else:
            delta_w2 = (k2 * error).unsqueeze(0)
            delta_w2 = self._clip_delta(delta_w2, self.max_last_layer_update_norm)
            self._last_linear.weight.add_(delta_w2.to(self._last_linear.weight.device))

        # Mild pullback towards checkpoint head to prevent cumulative drift.
        self._last_linear.weight.mul_(1.0 - self.anchor_decay_last).add_(
            self.anchor_decay_last * self._last_weight_anchor.to(self._last_linear.weight.device)
        )
        if self._use_bias_last and self._last_bias_anchor is not None:
            self._last_linear.bias.mul_(1.0 - self.anchor_decay_last).add_(
                self.anchor_decay_last * self._last_bias_anchor.to(self._last_linear.bias.device)
            )

        if scope == "last_layer":
            return

        # ---- Update hidden layer (W1, b1) via credit propagation ----
        x_flat = x_fusion.squeeze(0).to(self.device)
        phi1 = torch.cat([x_flat, x_flat.new_ones(1)]) if self._use_bias_first else x_flat

        self._P1, k1 = self._sm_update(self._P1, phi1)

        # Hidden credit uses local linearization through softplus and ReLU.
        # For scalar output: credit_i ≈ w2_i * (error * gain) * relu_mask_i.
        # Use W2 before its own update for a cleaner local linearization.
        credit = w2_pre[0] * (error * gain) * relu_mask.float().to(self.device)  # [256]

        if self._use_bias_first:
            delta_w1 = torch.outer(credit, k1[:-1])  # [256, fusion_dim]
            delta_w1 = self._clip_delta(delta_w1, self.max_first_layer_update_norm)
            self._first_linear.weight.add_(delta_w1.to(self._first_linear.weight.device))

            delta_b1 = credit * k1[-1]
            delta_b1 = self._clip_delta(delta_b1, self.max_first_layer_update_norm)
            self._first_linear.bias.add_(delta_b1.to(self._first_linear.bias.device))
        else:
            delta_w1 = torch.outer(credit, k1)
            delta_w1 = self._clip_delta(delta_w1, self.max_first_layer_update_norm)
            self._first_linear.weight.add_(delta_w1.to(self._first_linear.weight.device))

        self._first_linear.weight.mul_(1.0 - self.anchor_decay_first).add_(
            self.anchor_decay_first * self._first_weight_anchor.to(self._first_linear.weight.device)
        )
        if self._use_bias_first and self._first_bias_anchor is not None:
            self._first_linear.bias.mul_(1.0 - self.anchor_decay_first).add_(
                self.anchor_decay_first * self._first_bias_anchor.to(self._first_linear.bias.device)
            )
