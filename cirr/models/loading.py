"""Helpers for loading Forecaster models from checkpoint files.

Every script that evaluates or fine-tunes the model starts with the same
boilerplate:  instantiate ``Forecaster``, ``torch.load``, ``load_state_dict``.
This module centralises that logic so the scripts only need a single call.
"""

from pathlib import Path

import torch

from cirr.models.forecaster import Forecaster

# Default hyper-parameters used during training (fallback when the checkpoint
# does not store ``hyper_parameters``).
_DEFAULT_HPARAMS: dict = {
    "model_name": "timm-resnet50",
    "pretrained": False,
    "input_channels": 3,
    "loss_function": "GNLL",
    "lr": 1e-3,
    "lr_patience": 5,
}


def load_forecaster_from_checkpoint(
    checkpoint_path: Path,
    device: str = "cpu",
    use_hparams: bool = False,
) -> Forecaster:
    """Instantiate a :class:`Forecaster` and load weights from *checkpoint_path*.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the ``.ckpt`` checkpoint file produced by PyTorch Lightning.
    device : str
        Target device (``"cuda"`` or ``"cpu"``).
    use_hparams : bool
        When ``True`` the model is constructed from the ``hyper_parameters``
        stored inside the checkpoint (useful when the architecture may differ
        from the hard-coded defaults, as in :mod:`fl_benchmark`).  When
        ``False`` the default hyper-parameters are used.

    Returns
    -------
    model : Forecaster
        Model with weights loaded, placed on *device*, **not** in eval mode
        (the caller decides the mode).
    """
    data = torch.load(checkpoint_path, map_location=device)

    hp = data.get("hyper_parameters", _DEFAULT_HPARAMS) if use_hparams else _DEFAULT_HPARAMS

    model = Forecaster(
        model_name=hp.get("model_name", "timm-resnet50"),
        pretrained=hp.get("pretrained", False),
        input_channels=hp.get("input_channels", 3),
        loss_function=hp.get("loss_function", "GNLL"),
        lr=hp.get("lr", 1e-3),
        lr_patience=hp.get("lr_patience", 5),
    ).to(device)

    model.load_state_dict(data["state_dict"], strict=False)
    return model
