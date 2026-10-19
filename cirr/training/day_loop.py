"""Shared per-day predict/train loops for local and federated continual learning."""

from __future__ import annotations

from collections.abc import Callable
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cirr.fl.kc_prototypes import DEFAULT_KC_EDGES, make_regime_labeller
from cirr.models.forecaster import Forecaster

# Default upper bound on the adaptive task-loss/prototype-loss ratio (see its use
# in `train_day`).  0 disables the cap, which is the default.
#
# A cap of 100 was in force from 2026-09-17 to 2026-09-19 and silently switched the
# alignment term off: the natural scale is task_loss/avg_proto, and with embeddings
# sitting ~1e-4 from their class mean that ratio is 4e6-1e7, so the cap bound on
# every batch and cut the term to ~1e-6 of the task loss instead of the documented
# lambda (0.1 = 10 %).  Measured gradient contribution fell to 2.5e-7 of the task
# gradient, which is why own / shuffled / random / lambda x30 were all
# indistinguishable.  Gradient clipping (--adapt-max-grad-norm, 1.0 by default)
# already bounds the update, so the cap's stated purpose is covered without it.
_DEFAULT_PROTO_SCALE_CAP = 0.0

# Alignment objectives.
#   mse           — the original: pull every embedding onto its regime centroid.
#                   Minimised by making all embeddings identical, and the target is
#                   the model's own (shrinking) class mean recomputed each round, so
#                   nothing anchors against that.  Measured outcome: feature norm
#                   fell 1.8x, mean pairwise cosine rose 0.88 -> 0.999, within-regime
#                   spread fell 43x, and the dominant feature direction stopped
#                   tracking irradiance (corr 0.69 -> 0.16).  Representation collapse.
#   prototypical  — cosine similarity of each regime's batch CENTROID against ALL
#                   regime prototypes, through a cross-entropy.  Two differences that
#                   matter: it is computed on centroids, so within-regime spread (the
#                   signal a continuous target needs) is left free; and it pushes away
#                   from other regimes, so the collapsed solution maximises the loss
#                   instead of minimising it.  Bounded and scale-free, so lambda has a
#                   stable meaning and needs no adaptive rescaling.
ALIGN_LOSSES = ("mse", "prototypical")
_DEFAULT_ALIGN_TEMPERATURE = 0.1


def predict_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    device: str,
    batch_size: int,
    client_name: str = "",
) -> tuple[np.ndarray, np.ndarray, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Run batched inference for every sample in *day_indices*."""
    if not day_indices:
        return np.array([]), np.array([]), []

    model.eval()
    preds, tgts = [], []
    day_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        _pbar = tqdm(
            total=len(day_indices),
            desc=f"  [{client_name}] load+infer",
            leave=False,
            unit="smp",
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for start in range(0, len(day_indices), batch_size):
            batch_idx = day_indices[start : start + batch_size]
            imgs, hists, targets = [], [], []
            for idx in batch_idx:
                images, irr_hist, ghi_clear, target, _ = dataset[idx]
                imgs.append(images)
                hists.append(irr_hist)
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target)
                targets.append(target)
                day_samples.append((images, irr_hist, ghi_clear, target))
                _pbar.update(1)
            imgs_t = torch.stack(imgs).to(device)
            hists_t = torch.stack(hists).to(device)
            mean, _ = model(imgs_t, hists_t)
            preds.append(mean.cpu().numpy().ravel())
            tgts.extend(t.item() for t in targets)
        _pbar.close()
    return (
        np.concatenate(preds) if preds else np.array([]),
        np.array(tgts),
        day_samples,
    )


def train_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    batch_size: int,
    epochs: int,
    device: str,
    replay_entries: list[tuple[Any, int]] | None = None,
    global_state_dict: dict[str, Any] | None = None,
    mu: float = 0.0,
    rng_seed: int | None = None,
    day_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    client_name: str = "",
    global_prototypes: dict[int, torch.Tensor] | None = None,
    proto_lambda: float = 0.0,
    proto_scale_cap: float = _DEFAULT_PROTO_SCALE_CAP,
    align_loss: str = "mse",
    align_temperature: float = _DEFAULT_ALIGN_TEMPERATURE,
    classify_fn: Callable[[float, float], int] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra_loss_fn: Callable[[torch.Tensor, Forecaster], torch.Tensor] | None = None,
    shuffle_each_epoch: bool = True,
) -> float:
    """Train *model* for *epochs* on *day_indices* with optional replay/prox/proto hooks."""
    if not day_indices:
        return 0.0

    if day_samples is not None:
        preloaded_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = list(day_samples)
        if replay_entries:
            for ds, idx in tqdm(
                replay_entries,
                desc=f"  [{client_name}] load replay",
                leave=False,
                unit="smp",
                file=sys.stdout,
                dynamic_ncols=True,
            ):
                img, hist, _ghi, tgt, _ = ds[idx]
                if not isinstance(tgt, torch.Tensor):
                    tgt = torch.tensor(tgt)
                preloaded_pool.append((img, hist, _ghi, tgt))
        pool_base: list = preloaded_pool
        preloaded = True
    else:
        current_ds: list[tuple[Any, int]] = [(dataset, idx) for idx in day_indices]
        pool_base = current_ds + (replay_entries or [])
        preloaded = False

    model.train()

    prox_ref: dict[str, torch.Tensor] | None = None
    if mu > 0.0 and global_state_dict is not None:
        prox_ref = {k: v.detach().to(device) for k, v in global_state_dict.items() if v.is_floating_point()}

    use_proto = global_prototypes is not None and proto_lambda > 0.0
    proto_on_device: dict[int, torch.Tensor] = {}
    if use_proto:
        proto_on_device = {cls: proto.detach().to(device) for cls, proto in global_prototypes.items()}
        if classify_fn is None:
            # Same default as prototype extraction, so the alignment target of a
            # sample is always the prototype of the regime it was counted into.
            classify_fn = make_regime_labeller("proxy", DEFAULT_KC_EDGES)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    total_loss = 0.0
    count = 0
    rng = random.Random(rng_seed)
    pool = list(pool_base)
    rng.shuffle(pool)
    fixed_batches = [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]

    for ep in range(epochs):
        batches = fixed_batches
        if shuffle_each_epoch:
            ep_pool = list(pool_base)
            rng.shuffle(ep_pool)
            batches = [ep_pool[i : i + batch_size] for i in range(0, len(ep_pool), batch_size)]

        _pbar = tqdm(
            batches,
            desc=f"  [{client_name}] train e{ep + 1}/{epochs}",
            leave=False,
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for batch in _pbar:
            if preloaded:
                imgs = torch.stack([item[0] for item in batch]).to(device)
                hist = torch.stack([item[1] for item in batch]).to(device)
                tgts = torch.stack([item[3] for item in batch]).to(device)
                ghis = [float(item[2]) for item in batch] if use_proto else None
            else:
                rows = [ds[idx] for ds, idx in batch]  # type: ignore[index]
                imgs = torch.stack([r[0] for r in rows]).to(device)
                hist = torch.stack([r[1] for r in rows]).to(device)
                tgts = torch.stack([r[3] for r in rows]).to(device)
                ghis = [float(r[2]) for r in rows] if use_proto else None

            optimizer.zero_grad()

            if use_proto:
                mean, var, img_feats = model.forward_with_features(imgs, hist)
            else:
                mean, var = model(imgs, hist)
                img_feats = None

            if var is not None:
                loss = model.loss_fn(mean, tgts, var)
            elif model.loss_name == "Berhu":
                loss = model.loss_fn(mean * 1000, tgts * 1000)
            else:
                loss = model.loss_fn(mean, tgts)

            if prox_ref is not None:
                prox = sum(
                    ((p - prox_ref[n]) ** 2).sum() for n, p in model.named_parameters() if p.requires_grad and n in prox_ref
                )
                loss = loss + (mu / 2.0) * prox

            if use_proto and img_feats is not None:
                classes = [classify_fn(float(tgts[b].item()), (ghis or [0.0])[b]) for b in range(imgs.size(0))]
                if align_loss == "prototypical":
                    # Centroid-level prototypical cross-entropy.  Each regime present in
                    # the batch contributes one centroid, scored against every available
                    # prototype: the own-regime term pulls, the rest push.  Needs at
                    # least two prototypes for the push to mean anything.
                    proto_ids = sorted(proto_on_device)
                    present = [c for c in sorted(set(classes)) if c in proto_on_device]
                    if len(proto_ids) >= 2 and present:
                        cents = torch.stack(
                            [
                                F.normalize(
                                    img_feats[[b for b, c in enumerate(classes) if c == cls_id]].mean(0),
                                    dim=0,
                                )
                                for cls_id in present
                            ]
                        )
                        bank = F.normalize(torch.stack([proto_on_device[c] for c in proto_ids]), dim=1)
                        logits = cents @ bank.T / align_temperature
                        target = torch.tensor([proto_ids.index(c) for c in present], device=device, dtype=torch.long)
                        # Scale by the detached task loss so lambda keeps its
                        # documented meaning (a fraction of the task-loss magnitude)
                        # and transfers across sites whose error scales differ.  Unlike
                        # the mse variant this divides by nothing, so there is no
                        # blow-up as embeddings approach their prototype: the
                        # cross-entropy is bounded.
                        loss = loss + proto_lambda * loss.detach() * F.cross_entropy(logits, target)
                else:
                    proto_loss = torch.tensor(0.0, device=device)
                    n_proto = 0
                    for b in range(imgs.size(0)):
                        cls = classes[b]
                        # cls < 0 marks a sample with no usable clear-sky value; it is
                        # trained on as usual but pulled toward no prototype.
                        if cls >= 0 and cls in proto_on_device:
                            proto_loss = proto_loss + F.mse_loss(img_feats[b], proto_on_device[cls])
                            n_proto += 1
                    if n_proto > 0:
                        avg_proto = proto_loss / n_proto
                        scale = loss.detach() / (avg_proto.detach() + 1e-8)
                        if proto_scale_cap > 0:
                            scale = scale.clamp(max=proto_scale_cap)
                        loss = loss + proto_lambda * scale * avg_proto

            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(loss, model)

            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            count += 1

    return total_loss / count if count > 0 else 0.0
