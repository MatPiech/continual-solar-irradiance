"""Online gradient descent helpers for forecaster adaptation."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from cirr.losses import BerHuLoss
from cirr.models.forecaster import Forecaster

TrainScope = Literal["head_only", "head_and_encoder"]
Algorithm1Scope = Literal["none", "head_only", "head_and_encoder"]
# "none" keeps the encoder frozen, so head_and_encoder trains exactly the head.
# It is the default because it is what the published Chapter 4 campaign actually
# did — see the note on set_trainable.
EncoderUnfreeze = Literal["none", "last_blocks", "full"]


def pred_mean(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def make_regression_loss(name: str) -> nn.Module:
    if name == "berhu":
        return BerHuLoss(threshold=30, scale=1000)
    if name == "huber":
        return nn.HuberLoss()
    if name == "mae":
        return nn.L1Loss()
    return nn.MSELoss()


def algorithm1_scope(delayed_abs_error: float, t1: float, t2: float) -> Algorithm1Scope:
    if delayed_abs_error < t1:
        return "none"
    if delayed_abs_error < t2:
        return "head_only"
    return "head_and_encoder"


def set_trainable(
    model: Forecaster,
    scope: TrainScope,
    *,
    encoder_unfreeze: EncoderUnfreeze = "none",
) -> None:
    """Select the trainable parameter subspace.

    On the encoder branch, ``encoder_unfreeze`` decides how much of the backbone
    the high-error action reaches:

    ``none``
        Encoder stays frozen, so ``head_and_encoder`` trains exactly the head.
    ``last_blocks``
        The last two *parameterised* children of the encoder — for ResNet50, the
        layer3 and layer4 stages.
    ``full``
        The whole encoder, the S = R^P action the thesis text describes.

    Note on the default.  Until this fix, ``last_blocks`` took the last two
    children of the encoder unconditionally.  For a timm backbone built with
    ``num_classes=0`` those are ``SelectAdaptivePool2d`` and ``Identity``, which
    hold no parameters, so the high-error action trained the head and nothing
    else — identical to the low-error action.  Every published Chapter 4 number
    was produced under that behaviour, which is why ``none`` is the default:
    it names what those runs did instead of silently changing the method.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.network_head.parameters():
        param.requires_grad_(True)
    if scope == "head_and_encoder":
        if encoder_unfreeze == "full":
            for param in model.image_encoder.parameters():
                param.requires_grad_(True)
        elif encoder_unfreeze == "last_blocks":
            parameterised = [c for c in model.image_encoder.children() if any(p.numel() for p in c.parameters())]
            for child in parameterised[-2:]:
                for param in child.parameters():
                    param.requires_grad_(True)


def freeze_all(model: Forecaster) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _slope_penalty_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    *,
    volatility_multiplier: float = 0.01,
    scale_factor: float = 1000.0,
) -> torch.Tensor:
    current_irr = history[-1]
    true_delta = (target.view(-1)[0] - current_irr) * scale_factor
    pred_delta = (pred.view(-1)[0] - current_irr) * scale_factor
    slope_error = torch.abs(pred_delta - true_delta)
    hist_var = torch.var(history * scale_factor, unbiased=False)
    dynamic_weight = 1.0 + (hist_var.detach() * volatility_multiplier)
    return (slope_error * dynamic_weight) / scale_factor


@torch.enable_grad()
def ogd_step(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    scope: TrainScope,
    loss_fn: nn.Module,
    image: torch.Tensor,
    history: torch.Tensor,
    gt: torch.Tensor,
    device: str,
    *,
    slope_penalty: bool = False,
    volatility_multiplier: float = 0.01,
    max_grad_norm: float = 1.0,
    encoder_unfreeze: EncoderUnfreeze = "none",
) -> None:
    set_trainable(model, scope, encoder_unfreeze=encoder_unfreeze)
    model.eval()
    imgs = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
    hist = history.unsqueeze(0).to(device) if history.dim() == 1 else history.to(device)
    target = gt.to(device)
    optimizer.zero_grad()
    pred = pred_mean(model(imgs, hist))
    loss = loss_fn(pred, target.view_as(pred))
    if slope_penalty:
        loss = loss + _slope_penalty_loss(pred, target, history.to(device), volatility_multiplier=volatility_multiplier)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    freeze_all(model)


@torch.enable_grad()
def head_ogd_step(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    image: torch.Tensor,
    history: torch.Tensor,
    target: torch.Tensor,
    device: str,
) -> None:
    set_trainable(model, "head_only")
    model.train()
    optimizer.zero_grad()
    pred = pred_mean(model(image.unsqueeze(0).to(device), history.unsqueeze(0).to(device)))
    loss = loss_fn(pred, target.to(device))
    loss.backward()
    optimizer.step()
    model.eval()
    freeze_all(model)


@torch.enable_grad()
def sliding_minibatch_step(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    scope: TrainScope,
    loss_fn: nn.Module,
    buffer: list[dict[str, Any]],
    device: str,
) -> None:
    set_trainable(model, scope)
    model.eval()
    images = torch.stack([item["image"] for item in buffer]).to(device)
    histories = torch.stack([item["history"] for item in buffer]).to(device)
    targets = torch.cat([item["gt"] for item in buffer], dim=0).to(device)
    optimizer.zero_grad()
    pred = pred_mean(model(images, histories))
    loss = loss_fn(pred, targets.view_as(pred))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    freeze_all(model)
