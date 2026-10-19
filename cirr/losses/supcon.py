"""Physically-guided supervised contrastive loss (CSI-binned positives)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss with CSI-bin labels."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss.

        Parameters
        ----------
        features : Tensor [B, D]
            L2-normalized or raw embeddings (normalized internally).
        labels : Tensor [B]
            CSI bin indices.
        """
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        batch_size = features.shape[0]
        if batch_size < 2:
            return features.sum() * 0.0

        features = F.normalize(features, dim=1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Stabilize softmax
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_pos = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(mask_pos, min=1.0)
        loss = -mean_log_prob_pos
        valid = mask_pos > 0
        if not valid.any():
            return features.sum() * 0.0
        return loss[valid].mean()
