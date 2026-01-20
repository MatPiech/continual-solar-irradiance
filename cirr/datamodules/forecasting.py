from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch.utils.data
from lightning import LightningDataModule

from cirr.datamodules.datasets import FolsomDataset


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

        self._transforms = A.ReplayCompose([
            A.Resize(image_size[1], image_size[0]),
        ])
        self._augmentations = A.ReplayCompose([
            # transforms
            A.Resize(image_size[1], image_size[0]),
            # geometry augmentations
            A.Affine(rotate=(-10, 10), translate_percent=(-10, 10), scale=(0.9, 1.1)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ])

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage: str) -> None:
        with self._periods_path.open('rb') as f:
            periods = pd.read_pickle(f)

        test_periods = list(filter(lambda p: p['history'][-1]['image_name'].startswith('2016'), periods))
        train_val_periods = list(filter(lambda p: not p['history'][-1]['image_name'].startswith('2016'), periods))

        # number of days for validatation dataset to get 80-20 ratio of train-val datasets
        val_size = 146
        np.random.seed(self._seed)
        val_dates = [str(y) + str(m).zfill(2) + str(d).zfill(2) for y, m, d in zip(
            np.random.randint(2014, 2016, size=val_size),
            np.random.randint(1, 13, size=val_size),
            np.random.randint(1, 29, size=val_size),
        )]
        val_periods = list(filter(lambda p: p['history'][-1]['image_name'][:8] in val_dates, train_val_periods))
        train_periods = list(filter(lambda p: p['history'][-1]['image_name'][:8] not in val_dates, train_val_periods))

        self._train_dataset = FolsomDataset(
            data_root=self._data_root,
            periods=train_periods,
            transforms=self._augmentations if self._augment else self._transforms,
            image_size=self._image_size,
            image_mean=self._image_mean,
            image_std=self._image_std,
        )
        self._val_dataset = FolsomDataset(
            data_root=self._data_root,
            periods=val_periods,
            transforms=self._transforms,
            image_size=self._image_size,
            image_mean=self._image_mean,
            image_std=self._image_std,
        )
        self._test_dataset = FolsomDataset(
            data_root=self._data_root,
            periods=test_periods,
            transforms=self._transforms,
            image_size=self._image_size,
            image_mean=self._image_mean,
            image_std=self._image_std,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_dataset, batch_size=self._batch_size, num_workers=self._workers,
            pin_memory=True, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_dataset, batch_size=self._batch_size, num_workers=self._workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test_dataset, batch_size=self._batch_size, num_workers=self._workers,
            pin_memory=True
        )
