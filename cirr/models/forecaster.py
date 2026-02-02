import lightning.pytorch as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class Forecaster(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        input_channels: int,
        loss_function: str,
        lr: float,
        lr_patience: int,
    ):
        super().__init__()

        self._lr = lr
        self._lr_patience = lr_patience
        self.loss_name = loss_function

        # 1. Flag to check if we are doing Uncertainty Quantification
        self.is_probabilistic = loss_function == "GNLL"

        # 2. Image Encoder
        self.image_encoder = timm.create_model(
            model_name.replace("timm-", ""),
            pretrained=pretrained,
            num_classes=0,  # num_classes=0 removes the final classification layer automatically in newer timm versions
            in_chans=input_channels,
        )

        # Handling timm version differences for retrieving feature dim
        if hasattr(self.image_encoder, "num_features"):
            self.num_features = self.image_encoder.num_features
        else:
            # Fallback for some models
            self.num_features = self.image_encoder.fc.in_features
            self.image_encoder.fc = torch.nn.Identity()

        # 3. Dynamic Head Construction
        # If Probabilistic (GNLL), we need 2 outputs [Mean, Var].
        # For MSE/MAE, we only need 1 [Mean].
        output_dim = 2 if self.is_probabilistic else 1

        fusion_input_dim = self.num_features + 4  # Image features + 4 history values

        self.network_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

        # 4. Initialize Loss & Metrics
        self.loss_fn = self._configure_loss(loss_function)

        # Standard metrics for solar forecasting
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MeanSquaredError(),
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.MeanAbsolutePercentageError(),
            ]
        )
        self.train_metrics = metrics.clone("train_")
        self.val_metrics = metrics.clone("val_")
        self.test_metrics = metrics.clone("test_")

        self.save_hyperparameters()

    def _configure_loss(self, loss_name):
        """Factory method to get the correct loss function."""
        if loss_name == "GNLL":
            return nn.GaussianNLLLoss()
        elif loss_name == "MSE":
            return nn.MSELoss()
        elif loss_name == "MAE":
            return nn.L1Loss()
        elif loss_name == "SmoothL1":
            return nn.SmoothL1Loss()
        elif loss_name == "Huber":
            return nn.HuberLoss(delta=0.1)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def forward(self, x: torch.Tensor, irradiance_history: torch.Tensor):
        """
        Unified forward pass.
        Returns: (mean, variance)
        Note: 'variance' will be None if using MSE/MAE.
        """
        # 1. Encode Image
        img_feats = self.image_encoder(x)

        # 2. Fuse with History
        fused = torch.cat([img_feats, irradiance_history], dim=1)

        # 3. Predict
        out = self.network_head(fused)

        if self.is_probabilistic:
            # Split: [Batch, 2] -> Mean [Batch, 1], Variance [Batch, 1]
            mean = out[:, 0].unsqueeze(1)
            # Softplus + epsilon ensures variance > 0
            log_var = out[:, 1].unsqueeze(1)
            variance = F.softplus(log_var) + 1e-6
            return mean, variance
        else:
            # Standard Regression: [Batch, 1]
            return out, None

    def _shared_step(self, batch, stage):
        """
        Handles the logic for Train, Val, and Test to avoid code duplication.
        """
        source_images, source_irradiances, target_irradiances, _ = batch

        # Forward pass
        pred_mean, pred_var = self(source_images, source_irradiances)

        # Compute Loss
        if self.is_probabilistic:
            loss = self.loss_fn(pred_mean, target_irradiances, pred_var)
            # Log uncertainty only if available and trainer is registered
            if hasattr(self, "_trainer") and self._trainer is not None:
                self.log(f"{stage}_uncertainty_mean", pred_var.mean(), on_epoch=True, sync_dist=True)
        else:
            loss = self.loss_fn(pred_mean, target_irradiances)

        # Compute Standard Metrics (Accuracy)
        # Metrics always compare (Prediction vs Target), ignoring variance
        if stage == "train":
            self.train_metrics(pred_mean, target_irradiances)
            if hasattr(self, "_trainer") and self._trainer is not None:
                self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True)
        elif stage == "val":
            self.val_metrics(pred_mean, target_irradiances)
            if hasattr(self, "_trainer") and self._trainer is not None:
                self.log_dict(self.val_metrics, on_step=False, on_epoch=True, sync_dist=True)
        elif stage == "test":
            self.test_metrics(pred_mean, target_irradiances)
            if hasattr(self, "_trainer") and self._trainer is not None:
                self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True)

        if hasattr(self, "_trainer") and self._trainer is not None:
            self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self._lr_patience, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Ensure this matches the log name in validation_step
                "frequency": 1,
            },
        }
