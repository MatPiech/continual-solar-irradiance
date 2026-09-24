from pathlib import Path
from typing import Any

from albumentations import ReplayCompose
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image

from cirr.data.transforms import fuse_optical_flow, load_and_transform_image, normalize_optical_flow
from cirr.physics.clear_sky import resolve_clear_sky_value
from cirr.physics.sun import SITES, site_name, solar_angles, sun_mask

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
        physics: bool = False,
        crop_size: int | None = None,
    ):
        self._data_root = data_root
        self._periods = periods
        self._transforms = transforms
        self._image_size = image_size
        self._image_mean = np.array(image_mean, dtype=np.float32)
        self._image_std = np.array(image_std, dtype=np.float32)
        self._clear_sky_lookup = clear_sky_lookup or {}
        self._optical_flow = normalize_optical_flow(optical_flow)
        self._physics = physics
        self._crop_size = crop_size
        self._dis = None  # created lazily: cv2 objects do not survive DataLoader worker pickling
        if physics:
            site = site_name(data_root)
            self._sun_cfg = SITES[site]
            if self._sun_cfg["offset"] is None:
                raise ValueError(f"No fitted sun-mask camera parameters for site {site!r} (scripts/fit_sun_camera.py)")
            self._sun_angles = solar_angles(site, [p["history"][-1]["image_name"] for p in periods])

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

    def _load_physics_tensor(self, history: list[dict[str, Any]]) -> torch.Tensor:
        """IEP 7-channel input: RGB, sun mask x I_t, Norm B/R cloud, DIS flow t-5 -> t.

        Best configuration of the IEP paper.  Deviations from the paper code, on
        purpose: the flow is computed from scratch (the paper's DIS warm start was
        never enabled), and the sun channel carries I_t in kW/m^2, the scale the
        model's irradiance history already uses.  Image t-5 is augmented with the
        same replay as image t, so augmentation never shows up as motion.
        """
        name = history[-1]["image_name"]
        raw = np.asarray(Image.open(self._data_root / "images" / name))
        prev_raw = np.asarray(Image.open(self._data_root / "images" / history[-2]["image_name"]))
        zenith, azimuth = self._sun_angles[name]
        mask = sun_mask(raw.shape[:2], self._crop_size, zenith, azimuth, self._sun_cfg)

        out = self._transforms(image=raw, mask=mask)
        image, sun = out["image"], out["mask"].astype(np.float32)
        prev = self._transforms.replay(out["replay"], image=prev_raw)["image"]

        inside = self._crop_mask[..., 0].astype(bool)
        rgb = np.where(self._crop_mask, (image / 255.0 - self._image_mean) / self._image_std, 0.0)

        sun *= max(history[-1]["irradiance"], 0.0) / 1000.0

        # Normalised blue-red ratio (B-R)/(B+R), scaled to [0, 1] as in the paper code.
        r = image[..., 0].astype(np.float32)
        b = image[..., 2].astype(np.float32)
        nbr = np.where(inside, (b - r) / np.maximum(b + r, 1.0), 0.0)
        cloud = (nbr + 1.0) * 257.0 / 512.0

        if self._dis is None:
            cv2.setNumThreads(1)  # one thread per DataLoader worker; the default oversubscribes the CPU ~30x
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        flow = self._dis.calc(cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY), cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)

        stacked = np.concatenate([rgb, sun[..., None], cloud[..., None], flow], axis=2).astype(np.float32)
        return torch.from_numpy(stacked).permute(2, 0, 1)

    def __getitem__(self, index: int) -> ForecastSample:
        period = self._periods[index]
        history = period["history"]
        timestamp = history[-1]["image_name"][:-4]

        if self._physics:
            image_input = self._load_physics_tensor(history)
        else:
            image_input = self._load_image_tensor(history[-1]["image_name"])

        if not self._physics and self._optical_flow != "none" and len(history) >= 2:
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
