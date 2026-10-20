from pathlib import Path
import pickle

import albumentations as A
from lightning import LightningDataModule
import numpy as np
import torch.utils.data

from cirr.data.datasets import SkyImageDataset
from cirr.physics.clear_sky import load_clear_sky_lookup


class ForecastingDataModule(LightningDataModule):
    def __init__(
        self,
        root_data_path: Path,
        periods_path: Path,
        augment: bool,
        image_size: tuple[int, int],
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
        batch_size: int,
        workers: int,
        seed: int,
        optical_flow: str = "none",
        physics: bool = False,
    ):
        super().__init__()

        self._data_root = Path(root_data_path)
        self._periods_path = Path(periods_path)
        self._dataset_name = self._data_root.name
        self._augment = augment
        self._image_size = image_size
        self._image_mean = image_mean
        self._image_std = image_std
        self._batch_size = batch_size
        self._workers = workers
        self._seed = seed
        self._optical_flow = optical_flow
        self._physics = physics
        self._clear_sky_lookup = load_clear_sky_lookup(self._data_root)

        dataset_name_lower = self._dataset_name.lower()
        if any(name in dataset_name_lower for name in ("folsom", "modena", "poznan", "karpicko")):
            crop_size = 1520
        elif any(name in dataset_name_lower for name in ("bern", "alpnach", "ne")):
            crop_size = 1200
        else:
            raise ValueError(
                "Unsupported dataset name for crop size selection: "
                f"{self._dataset_name}. Expected name containing one of: "
                "folsom, modena, poznan, karpicko, bern, alpnach, ne."
            )

        self._crop_size = crop_size
        self._transforms = A.ReplayCompose(
            [
                A.CenterCrop(height=crop_size, width=crop_size),
                A.Resize(image_size[1], image_size[0]),
            ]
        )
        self._augmentations = A.ReplayCompose(
            [
                # transforms
                A.CenterCrop(height=crop_size, width=crop_size),
                A.Resize(image_size[1], image_size[0]),
                # geometry augmentations
                # translate_percent is a fraction of the image size: (-10, 10) shifted by up to 10x and
                # blanked ~50 % of training frames.  IEP paper: up to 10 %.
                A.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.9, 1.1)),
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ]
        )

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage: str) -> None:
        with self._periods_path.open("rb") as f:
            periods = pickle.load(f)

        if "folsom" in self._dataset_name.lower():
            test_year = "2016"
            test_periods = list(filter(lambda p: p["history"][-1]["image_name"].startswith(test_year), periods))
            train_val_periods = list(filter(lambda p: not p["history"][-1]["image_name"].startswith(test_year), periods))
            # number of days for validatation dataset to get 80-20 ratio of train-val datasets
            val_size = 146
            np.random.seed(self._seed)
            val_dates = [
                str(y) + str(m).zfill(2) + str(d).zfill(2)
                for y, m, d in zip(
                    np.random.randint(2014, 2016, size=val_size),
                    np.random.randint(1, 13, size=val_size),
                    np.random.randint(1, 29, size=val_size),
                    strict=False,
                )
            ]
            val_periods = list(filter(lambda p: p["history"][-1]["image_name"][:8] in val_dates, train_val_periods))
            train_periods = list(filter(lambda p: p["history"][-1]["image_name"][:8] not in val_dates, train_val_periods))
        else:
            test_periods = periods
            val_periods = []
            train_periods = []

        ds_kwargs = {
            "data_root": self._data_root,
            "image_size": self._image_size,
            "image_mean": self._image_mean,
            "image_std": self._image_std,
            "clear_sky_lookup": self._clear_sky_lookup,
            "optical_flow": self._optical_flow,
            "physics": self._physics,
            "crop_size": self._crop_size,
        }

        self._train_dataset = SkyImageDataset(
            periods=train_periods,
            transforms=self._augmentations if self._augment else self._transforms,
            **ds_kwargs,
        )
        self._val_dataset = SkyImageDataset(
            periods=val_periods,
            transforms=self._transforms,
            **ds_kwargs,
        )
        self._test_dataset = SkyImageDataset(
            periods=test_periods,
            transforms=self._transforms,
            **ds_kwargs,
        )

    @property
    def train_dataset(self) -> SkyImageDataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> SkyImageDataset:
        return self._test_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_dataset, batch_size=self._batch_size, num_workers=self._workers, pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test_dataset, batch_size=self._batch_size, num_workers=self._workers, pin_memory=True
        )
