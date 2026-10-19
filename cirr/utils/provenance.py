"""Run provenance: determinism, file hashes, git commit.

Chapters 4 and 5 record provenance the same way, so a result folder from either
campaign can be traced back to the same three facts: which commit produced it,
which checkpoint it loaded, and which seed it was pinned to.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch


def set_determinism(seed: int) -> None:
    """Pin every RNG the day loop touches, and ask torch for deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # warn_only: some timm kernels have no deterministic implementation; a warning
    # is more useful than refusing to run the benchmark at all.
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # provenance is best-effort, never fatal
        return "unknown"
