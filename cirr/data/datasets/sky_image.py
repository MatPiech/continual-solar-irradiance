from pathlib import Path
from typing import Any

from albumentations import ReplayCompose
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from cirr.data.transforms import fuse_optical_flow, load_and_transform_image, normalize_optical_flow
from cirr.physics.clear_sky import resolve_clear_sky_value

# Unified sample: (image, history, ghi_clear, target, timestamp)
ForecastSample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]


class SkyImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        periods: list[dict[str, Any]],
        transforms: ReplayCompose,
        image_size: tuple[int, int],
        image_mean: tuple[float, ...],
        image_std: tuple[float, ...],
        clear_sky_lookup: dict[str, float] | None = None,
        optical_flow: str = "none",
    ):
        self._data_root = data_root
        self._periods = periods
        self._transforms = transforms
        self._image_size = image_size
        self._image_mean = np.array(image_mean, dtype=np.float32)
        self._image_std = np.array(image_std, dtype=np.float32)
        self._clear_sky_lookup = clear_sky_lookup or {}
        self._optical_flow = normalize_optical_flow(optical_flow)

        self._crop_mask = cv2.circle(
            np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8),
            (image_size[1] // 2, image_size[0] // 2),
            image_size[0] // 2,
            color=(1, 1, 1),
            thickness=-1,
        )

    def _load_image_tensor(self, image_name: str) -> torch.Tensor:
        image_path = self._data_root / "images" / image_name
        cropped = load_and_transform_image(
            image_path,
            self._transforms,
            self._crop_mask,
            self._image_mean,
            self._image_std,
        )
        return torch.from_numpy(cropped).permute(2, 0, 1)

    def __getitem__(self, index: int) -> ForecastSample:
        period = self._periods[index]
        history = period["history"]

        image_input = self._load_image_tensor(history[-1]["image_name"])
        timestamp = history[-1]["image_name"][:-4]

        if self._optical_flow != "none" and len(history) >= 2:
            prev_image = self._load_image_tensor(history[-2]["image_name"])
            image_input = fuse_optical_flow(self._optical_flow, image_input, prev_image, image_input)

        source_irradiances = [item["irradiance"] / 1000.0 for item in history]
        target_irradiance = period["target_irradiance"] / 1000.0

        ghi_clear = resolve_clear_sky_value(self._clear_sky_lookup, timestamp)
        if ghi_clear is None:
            ghi_clear = 0.0

        return (
            image_input,
            torch.tensor(source_irradiances, dtype=torch.float32),
            torch.tensor(ghi_clear, dtype=torch.float32),
            torch.tensor([target_irradiance], dtype=torch.float32),
            timestamp,
        )

    def __len__(self) -> int:
        return len(self._periods)

    @property
    def periods(self) -> list[dict[str, Any]]:
        return self._periods
