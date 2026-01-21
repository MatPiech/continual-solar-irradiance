from pathlib import Path
from typing import Any

from albumentations import ReplayCompose
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class FolsomDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        periods: list[dict[str, Any]],
        transforms: ReplayCompose,
        image_size: tuple[int],
        image_mean: tuple[float],
        image_std: tuple[float],
    ):
        self._data_root = data_root
        self._periods = periods
        self._transforms = transforms
        self._image_size = image_size
        self._image_mean = image_mean
        self._image_std = image_std

        self._crop_mask = cv2.circle(
            np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8),
            (image_size[1] // 2, image_size[0] // 2),
            image_size[0] // 2,
            color=(1, 1, 1),
            thickness=-1,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period = self._periods[index]

        image_path = self._data_root / "images" / period["history"][-1]["image_name"]
        image = self._transforms(image=np.asarray(Image.open(image_path)))["image"]
        norm_image = (image / 255.0 - self._image_mean) / self._image_std
        cropped_image = np.where(self._crop_mask, norm_image, 0.0).astype(np.float32)
        image_input = torch.from_numpy(cropped_image).permute(2, 0, 1)

        source_irradiances = []
        for history_item in period["history"]:
            irradiance = history_item["irradiance"]
            source_irradiances.append(irradiance)

        target_irradiance = period["target_irradiance"]

        return (
            image_input,
            torch.Tensor(source_irradiances),
            torch.Tensor([target_irradiance]),
            period["history"][-1]["image_name"][:-4],
        )

    def __len__(self) -> int:
        return len(self._periods)
