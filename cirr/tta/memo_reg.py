"""MEMO-Reg: Test-Time Adaptation via Prediction Variance Minimization for Regression.

Adapts MEMO (Zhang et al., NeurIPS 2022) from classification to regression.
Standard MEMO minimises the marginal entropy of classification softmax outputs
across augmented views of a single test point.  For regression there is no
probability simplex, so entropy is ill-defined.  MEMO-Reg replaces the entropy
objective with *prediction variance minimisation*:

    Loss = (1/N) * Σ_i (ŷ_i − ȳ)²,   ȳ = (1/N) * Σ_i ŷ_i

Minimising this encourages the model to produce the *same* forecast regardless
of which augmentation of the test image is presented — a consistency proxy for
calibration under domain shift.

Procedure (one test step)
--------------------------
1. Given a single sky image *I* and its irradiance history *h*:
2. Generate *N* augmented views {I_1, …, I_N} via geometric & photometric ops.
3. Forward-pass each view through the model with the *same* irradiance history.
4. Compute the variance loss and take *K* gradient steps (typically K=1).
5. Forward-pass the **original unaugmented** image through the updated model
   and return its prediction as the final forecast.

References
----------
.. [1] Zhang, M., Levine, S., & Finn, c. (2022).
       "MEMO: Test Time Robustness via Adaptation and Augmentation."
       NeurIPS 2022. https://arxiv.org/abs/2110.09506
"""

from copy import deepcopy
from typing import Literal

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from cirr.tta.common import prediction_variance_loss, select_adapt_params

# ---------------------------------------------------------------------------
# Tensor-space augmentations (operate on normalised float tensors [c, h, w])
# ---------------------------------------------------------------------------


def _random_augment(
    img: torch.Tensor,
    max_rotation: float = 10.0,
    translate_percent: float = 0.10,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
) -> torch.Tensor:
    """Apply random geometric augmentations matching the training-time config.

    Mirrors the albumentations pipeline used in ``ForecastingDataModule``:

    .. code-block:: python

        A.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.9, 1.1))
        A.HorizontalFlip()
        A.VerticalFlip()

    All operations work directly on the normalised float tensor so that no
    de-normalisation / re-normalisation round-trip is required.

    Parameters
    ----------
    img : torch.Tensor
        Normalised image tensor of shape ``[c, h, w]``.
    max_rotation : float, optional
        Maximum rotation angle in degrees, by default 10  (matches training).
    translate_percent : float, optional
        Maximum translation as a fraction of image size, by default 0.10.
    scale_min : float, optional
        Minimum random scale factor, by default 0.9.
    scale_max : float, optional
        Maximum random scale factor, by default 1.1.

    Returns
    -------
    torch.Tensor
        Augmented image tensor of the same shape.
    """
    _, h, w = img.shape

    # Horizontal / vertical flip
    if torch.rand(1).item() > 0.5:
        img = TF.hflip(img)
    if torch.rand(1).item() > 0.5:
        img = TF.vflip(img)

    # Affine: rotation + translation + scale (matches albumentations Affine)
    angle = (torch.rand(1).item() * 2 - 1) * max_rotation
    tx = int((torch.rand(1).item() * 2 - 1) * translate_percent * w)
    ty = int((torch.rand(1).item() * 2 - 1) * translate_percent * h)
    scale = scale_min + torch.rand(1).item() * (scale_max - scale_min)
    img = TF.affine(
        img,
        angle=angle,
        translate=[tx, ty],
        scale=float(scale),
        shear=0.0,
        fill=0.0,
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    return img


# ---------------------------------------------------------------------------
# MEMO-Reg
# ---------------------------------------------------------------------------


class MEMOReg(nn.Module):
    """MEMO for regression: per-sample adaptation via prediction variance minimisation.

    At test time the model is momentarily updated to produce *consistent*
    predictions across *N* augmented views of the same sky image.  The
    (episodically) adapted model is then evaluated on the original unaugmented
    image to yield the final irradiance forecast.

    Parameters
    ----------
    model : nn.Module
        The ``Forecaster`` (or compatible) model to adapt.
    lr : float, optional
        Adaptation learning rate, by default 1e-4.
    weight_decay : float, optional
        Optimizer weight decay, by default 0.0.
    n_augmentations : int, optional
        Number of augmented views per test image, by default 8.
    niter : int, optional
        Number of gradient steps per test image, by default 1.
    episodic : bool, optional
        Reset weights before *each* test image.  Almost always True for
        MEMO-style adaptation, by default True.
    adapt_mode : {'all', 'bn_only'}, optional
        Which parameters to update.
        - ``'all'``: optimise the entire model.
        - ``'bn_only'``: optimise only BatchNorm affine parameters (γ, β).
        By default ``'all'``.
    max_rotation : float, optional
        Maximum random rotation in degrees, by default 10 (matches training).
    translate_percent : float, optional
        Maximum translation as fraction of image size, by default 0.10 (matches training).
    scale_min : float, optional
        Minimum scale factor for affine augmentation, by default 0.9 (matches training).
    scale_max : float, optional
        Maximum scale factor for affine augmentation, by default 1.1 (matches training).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        n_augmentations: int = 8,
        niter: int = 1,
        episodic: bool = True,
        adapt_mode: Literal["all", "bn_only"] = "all",
        max_rotation: float = 10.0,
        translate_percent: float = 0.10,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
    ):
        super().__init__()

        self.model = model
        self.episodic = episodic

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_augmentations = n_augmentations
        self.niter = niter
        self.adapt_mode = adapt_mode
        self.max_rotation = max_rotation
        self.translate_percent = translate_percent
        self.scale_min = scale_min
        self.scale_max = scale_max

        # Save initial weights for episodic reset
        self._initial_model_state = deepcopy(model.state_dict())

        # Validate and configure
        self.check_model()
        self.configure_model()

        # Optimizer (created after configure_model sets requires_grad)
        self.optimizer = torch.optim.Adam(select_adapt_params(model, adapt_mode), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # Public interface override: regression takes a 2-tuple
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        irradiance_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Test-time forward pass with per-image MEMO-Reg adaptation.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images, shape ``[batch_size, c, h, w]``.
        irradiance_history : torch.Tensor
            Batch of irradiance history vectors, shape ``[batch_size, history_len]``.

        Returns
        -------
        means : torch.Tensor
            Predicted mean irradiance values, shape ``[batch_size, 1]``.
        variances : torch.Tensor or None
            Predicted aleatoric variances from GNLL head, or ``None``.
        """
        if self.episodic:
            self.reset()

        return self.adapt(images, irradiance_history)

    # ------------------------------------------------------------------
    # adapt() – core MEMO-Reg logic
    # ------------------------------------------------------------------

    def adapt(
        self,
        images: torch.Tensor,
        irradiance_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Adapt the model for each image in the batch and return predictions.

        Each image is processed independently: the model is updated to minimise
        prediction variance across augmented views, then the original image is
        forwarded for the final forecast.

        Parameters
        ----------
        images : torch.Tensor
            ``[batch_size, c, h, w]`` batch.
        irradiance_history : torch.Tensor
            ``[batch_size, history_len]`` batch.

        Returns
        -------
        means : torch.Tensor
            Shape ``[batch_size, 1]``.
        variances : torch.Tensor or None
            Shape ``[batch_size, 1]`` or ``None``.
        """
        batch_size = images.size(0)
        device = images.device
        means_list = []
        vars_list = []

        for i in range(batch_size):
            img_i = images[i]  # [c, h, w]
            hist_i = irradiance_history[i].unsqueeze(0)  # [1, history_len]

            # Reset to initial weights for this image (episodic per-sample)
            self.model.load_state_dict(self._initial_model_state, strict=True)
            self.configure_model()

            # Gradient adaptation step(s)
            self._adapt_single(img_i, hist_i, device)

            # Final prediction on original unaugmented image
            mean_i, var_i = self._predict_single(img_i, hist_i, device)
            means_list.append(mean_i)
            vars_list.append(var_i)

        means = torch.cat(means_list, dim=0)  # [batch_size, 1]
        variances = torch.cat(vars_list, dim=0) if vars_list[0] is not None else None
        return means, variances

    def reset(self) -> None:
        """Reset model and optimizer to initial state."""
        self.model.load_state_dict(self._initial_model_state, strict=True)
        # Re-register params after weight reset so optimizer sees fresh tensors
        params = select_adapt_params(self.model, self.adapt_mode)
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        self.configure_model()

    def configure_model(self) -> nn.Module:
        """Configure trainability according to ``adapt_mode``."""
        if self.adapt_mode == "bn_only":
            # Freeze everything, then enable only BN affine params
            for param in self.model.parameters():
                param.requires_grad_(False)
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d):
                    if m.affine:
                        m.weight.requires_grad_(True)
                        m.bias.requires_grad_(True)
                    # Use per-batch statistics (not running stats) during adaptation
                    m.train()
            # Put the rest in eval mode
            self.model.eval()
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d):
                    m.train()
        else:
            # 'all': enable gradients for every parameter
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad_(True)
        return self.model

    def check_model(self) -> None:
        """Validate model compatibility.

        For ``bn_only`` mode the model must contain at least one BatchNorm layer.
        """
        if self.adapt_mode == "bn_only":
            has_bn = any(isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d) for m in self.model.modules())
            if not has_bn:
                raise AssertionError(
                    "MEMOReg with adapt_mode='bn_only' requires BatchNorm layers. "
                    "Use adapt_mode='all' for models without BN."
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def _adapt_single(self, img: torch.Tensor, hist: torch.Tensor, device: torch.device) -> None:
        """Minimise prediction variance over augmented views of one image."""
        for _ in range(self.niter):
            # Build augmented batch [N, c, h, w] and replicated history [N, history_len]
            aug_imgs = torch.stack(
                [
                    _random_augment(
                        img,
                        self.max_rotation,
                        self.translate_percent,
                        self.scale_min,
                        self.scale_max,
                    )
                    for _ in range(self.n_augmentations)
                ],
                dim=0,
            ).to(device)  # [N, c, h, w]
            aug_hist = hist.expand(self.n_augmentations, -1)  # [N, history_len]

            self.optimizer.zero_grad()
            output = self.model(aug_imgs, aug_hist)

            # Handle probabilistic (GNLL) models: output = (mean, variance)
            preds = output[0] if isinstance(output, tuple) else output  # [N, 1]

            loss = prediction_variance_loss(preds)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def _predict_single(
        self,
        img: torch.Tensor,
        hist: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run inference on the original (unaugmented) test image."""
        self.model.eval()
        x = img.unsqueeze(0).to(device)  # [1, c, h, w]
        h = hist.to(device)  # [1, history_len]
        output = self.model(x, h)

        if isinstance(output, tuple):
            mean, variance = output  # [1, 1], [1, 1] or None
        else:
            mean, variance = output, None

        return mean, variance

    def __repr__(self) -> str:
        return (
            f"MEMOReg(n_augmentations={self.n_augmentations}, "
            f"niter={self.niter}, lr={self.lr}, adapt_mode={self.adapt_mode!r}, "
            f"max_rotation={self.max_rotation}, translate_percent={self.translate_percent}, "
            f"scale=[{self.scale_min},{self.scale_max}], episodic={self.episodic})"
        )
