"""Optical flow, early fusion, and input channel helpers."""

from __future__ import annotations

from pathlib import Path

from albumentations import ReplayCompose
import cv2
import numpy as np
from PIL import Image
import torch

RGB_CHANNELS = 3
FLOW_CHANNELS = 2


def normalize_optical_flow(value: str | bool | None) -> str:
    if value is None or value is False:
        return "none"
    if value is True:
        return "farneback"
    method = str(value).strip().lower()
    if method in ("none", "off", "false", "0", ""):
        return "none"
    if method == "farneback":
        return "farneback"
    raise ValueError(f"Unsupported optical_flow {value!r}. Use 'none' or 'farneback'.")


def input_channels(*, optical_flow: str | bool | None = "none") -> int:
    return RGB_CHANNELS + (FLOW_CHANNELS if normalize_optical_flow(optical_flow) != "none" else 0)


def load_and_transform_image(
    image_path: Path,
    transforms: ReplayCompose,
    crop_mask: np.ndarray,
    image_mean: np.ndarray,
    image_std: np.ndarray,
) -> np.ndarray:
    """Load, augment, normalize, and circular-crop a sky image."""
    image = transforms(image=np.asarray(Image.open(image_path)))["image"]
    norm_image = (image / 255.0 - image_mean) / image_std
    return np.where(crop_mask, norm_image, 0.0).astype(np.float32)


def compute_farneback_flow(prev_img: np.ndarray, curr_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute dense optical flow between two RGB images (H, W, 3) in [0, 1] scale."""
    prev_gray = cv2.cvtColor((prev_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor((curr_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow[..., 0], flow[..., 1]


def normalize_flow(u: np.ndarray, v: np.ndarray, scale: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Scale flow components to roughly ImageNet-normalized magnitude."""
    return u / scale, v / scale


def fuse_optical_flow(
    method: str,
    rgb: torch.Tensor,
    prev_rgb: torch.Tensor,
    curr_rgb: torch.Tensor | None = None,
    flow_scale: float = 0.05,
) -> torch.Tensor:
    """Fuse RGB with optical-flow channels for the requested method."""
    if method == "none":
        return rgb
    if method == "farneback":
        return early_fusion(rgb, prev_rgb, curr_rgb, flow_scale=flow_scale)
    raise ValueError(f"Unsupported optical_flow {method!r}. Use 'none' or 'farneback'.")


def early_fusion(
    rgb: torch.Tensor,
    prev_rgb: torch.Tensor,
    curr_rgb: torch.Tensor | None = None,
    flow_scale: float = 0.05,
) -> torch.Tensor:
    """Build 5-channel tensor [R, G, B, u, v] from current RGB and optical flow."""
    if curr_rgb is None:
        curr_rgb = rgb

    prev_np = prev_rgb.permute(1, 2, 0).cpu().numpy()
    curr_np = curr_rgb.permute(1, 2, 0).cpu().numpy()
    # Denormalize for flow — approximate inverse of ImageNet norm on [0,1] pixels
    prev_denorm = np.clip(prev_np * 0.229 + 0.485, 0.0, 1.0)
    curr_denorm = np.clip(curr_np * 0.229 + 0.485, 0.0, 1.0)

    u, v = compute_farneback_flow(prev_denorm, curr_denorm)
    u, v = normalize_flow(u, v, scale=flow_scale)
    flow = np.stack([u, v], axis=0).astype(np.float32)
    return torch.cat([rgb, torch.from_numpy(flow)], dim=0)


def _demo() -> None:
    assert input_channels(optical_flow="none") == 3
    assert input_channels(optical_flow="farneback") == 5
    h, w = 32, 32
    fused = early_fusion(torch.randn(3, h, w), torch.randn(3, h, w), torch.randn(3, h, w))
    assert fused.shape == (5, h, w)
    print("transforms self-check ok")


if __name__ == "__main__":
    _demo()
