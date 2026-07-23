"""Visualization utilities for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_embedding_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "Latent embeddings",
) -> None:
    """2D scatter via PCA (no extra deps beyond sklearn if available)."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("sklearn required for plot_embedding_scatter") from None

    xy = PCA(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.6, label=str(label))
    plt.legend(markerscale=2, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pareto_curve(
    backprop_fractions: list[float],
    mae_values: list[float],
    output_path: Path,
    title: str = "Accuracy vs compute",
) -> None:
    """Plot forecast accuracy (MAE) vs percentage of backprop rounds."""
    plt.figure(figsize=(7, 5))
    plt.plot(backprop_fractions, mae_values, "o-")
    plt.xlabel("Backprop fraction (%)")
    plt.ylabel("MAE")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_gradcam_rgb(
    heatmap: np.ndarray,
    rgb_image: np.ndarray,
    output_path: Path,
) -> None:
    """Overlay Grad-CAM heatmap on RGB channels of 5-ch input."""
    plt.figure(figsize=(5, 5))
    rgb = np.clip(rgb_image.transpose(1, 2, 0), 0, 1)
    plt.imshow(rgb)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
