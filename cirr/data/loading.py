"""Convenience helpers for constructing ForecastingDataModule instances."""

from pathlib import Path
from typing import Any

from cirr.data.forecasting import ForecastingDataModule

DEFAULT_IMAGE_SIZE: list[int] = [128, 128]
DEFAULT_IMAGE_MEAN: list[float] = [0.485, 0.456, 0.406]  # ImageNet mean
DEFAULT_IMAGE_STD: list[float] = [0.229, 0.224, 0.225]  # ImageNet std


def load_dataset(
    dataset_path: Path,
    split: str = "test",
    augment: bool = False,
    batch_size: int = 1,
    workers: int = 0,
    seed: int = 42,
    optical_flow: str = "none",
) -> Any:
    """Load a dataset split from a prepared directory."""
    if split == "test":
        periods_path = dataset_path / "periods.pickle"
    elif split == "training":
        periods_path = dataset_path / "train_periods.pickle"
    else:
        raise ValueError(f"Invalid split {split!r}. Choose 'test' or 'training'.")

    dm = ForecastingDataModule(
        root_data_path=dataset_path,
        periods_path=periods_path,
        augment=augment,
        image_size=DEFAULT_IMAGE_SIZE,
        image_mean=DEFAULT_IMAGE_MEAN,
        image_std=DEFAULT_IMAGE_STD,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        optical_flow=optical_flow,
    )
    dm.setup(None)

    if split == "test":
        return dm.test_dataset
    return dm.train_dataset
