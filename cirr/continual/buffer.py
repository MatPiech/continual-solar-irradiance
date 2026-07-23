"""FIFO delayed-feedback buffer for lagged ground-truth gating."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TypeVar

import torch

_T = TypeVar("_T")


def reveal_after_horizon(queue: deque[_T], horizon_steps: int, item: _T) -> _T | None:
    queue.append(item)
    if len(queue) > horizon_steps:
        return queue.popleft()
    return None


@dataclass
class DelayedState:
    timestamp: str
    z_t: torch.Tensor
    x_t: torch.Tensor
    y_hat: float
    ghi_clear: float
    csi_bin: int
    y_target: float
    history: torch.Tensor | None = None


class DelayedFeedbackBuffer:
    def __init__(self, horizon_steps: int = 3):
        self.horizon_steps = horizon_steps
        self._queue: deque[DelayedState] = deque()

    def enqueue(self, state: DelayedState) -> DelayedState | None:
        return reveal_after_horizon(self._queue, self.horizon_steps, state)

    def clear(self) -> None:
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)
