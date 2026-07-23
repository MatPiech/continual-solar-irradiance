"""Clear-Sky Index (CSI) computation and physical regime binning."""

from __future__ import annotations

# Default 3-bin CSI edges
CSI_OVERCAST = 0.3
CSI_CLEAR = 0.85

# 5-bin edges (optional ablation)
CSI_BINS_5 = (0.15, 0.3, 0.6, 0.85)


def compute_csi(y: float, ghi_clear: float, eps: float = 1e-3) -> float:
    """Compute CSI = y / GHI_clear with denominator clamping."""
    denom = max(float(ghi_clear), eps)
    return float(y) / denom


def discretize_csi(csi: float, n_bins: int = 3) -> int:
    """Map continuous CSI to a physical regime bin index.

    n_bins=3: 0=overcast (<0.3), 1=mixed [0.3, 0.85), 2=clear (>=0.85)
    n_bins=5: finer bins using CSI_BINS_5 cutpoints.
    """
    if n_bins == 3:
        if csi < CSI_OVERCAST:
            return 0
        if csi < CSI_CLEAR:
            return 1
        return 2

    if n_bins == 5:
        edges = CSI_BINS_5
        for idx, edge in enumerate(edges):
            if csi < edge:
                return idx
        return len(edges)

    raise ValueError(f"Unsupported n_bins={n_bins}. Use 3 or 5.")


def _demo() -> None:
    assert abs(compute_csi(0.5, 0.6) - 0.5 / 0.6) < 1e-6
    assert discretize_csi(0.1) == 0
    assert discretize_csi(0.5) == 1
    assert discretize_csi(0.9) == 2
    assert discretize_csi(0.2, n_bins=5) == 1
    print("csi self-check ok")


if __name__ == "__main__":
    _demo()
