"""Convenience helpers for constructing ForecastingDataModule instances.

All scripts share the same normalisation constants and image size.  This
module centralises those defaults so that changes propagate everywhere
without per-script edits.
"""

from pathlib import Path
from typing import Any

from cirr.datamodules.forecasting import ForecastingDataModule

# ---------------------------------------------------------------------------
# Shared defaults (match training-time configuration)
# ---------------------------------------------------------------------------

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
) -> Any:
    """Load a dataset split from a prepared directory.

    Parameters
    ----------
    dataset_path : Path
        Path to the prepared dataset directory (e.g. ``data/poznan_prepared``).
    split : {'test', 'training'}
        Which split to return.  ``'test'`` uses ``periods.pickle``;
        ``'training'`` uses ``train_periods.pickle``.
    augment : bool
        Whether to apply training-time augmentations, by default ``False``.
    batch_size : int
        DataModule batch size, by default ``1``.
    workers : int
        Number of DataLoader workers, by default ``0``.
    seed : int
        Random seed for the DataModule, by default ``42``.

    Returns
    -------
    dataset : SkyImageDataset
        The requested dataset object with a ``._periods`` attribute.

    Raises
    ------
    ValueError
        If *split* is not ``'test'`` or ``'training'``.
    """
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
    )
    dm.setup(None)

    if split == "test":
        return dm._test_dataset
    return dm._train_dataset
