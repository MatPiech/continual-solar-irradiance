"""Shared TTA helpers used by MEMO, NOTE, and dual-phase adaptation."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def prediction_variance_loss(predictions: torch.Tensor) -> torch.Tensor:
    """Mean prediction variance across augmented views."""
    preds = predictions.squeeze(-1)
    mean_pred = preds.mean()
    return ((preds - mean_pred) ** 2).mean()


def select_adapt_params(model: nn.Module, adapt_mode: Literal["all", "bn_only"]) -> list[nn.Parameter]:
    """Return optimizer parameters for TTA (all weights or BN affine only)."""
    if adapt_mode == "bn_only":
        params: list[nn.Parameter] = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d) and m.affine:
                params.extend([m.weight, m.bias])
        return params
    return list(model.parameters())
