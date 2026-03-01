import lightning.pytorch as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from cirr.losses import MeanAdaptiveBerHuLoss
from cirr.models.lora import LoRAConv2d, LoRALinear, count_lora_parameters, print_lora_info, replace_linear_with_lora


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
            num_classes=1,  # num_classes=0 removes the final classification layer automatically in newer timm versions
            in_chans=input_channels,
        )

        # Handling timm version differences for retrieving feature dim
        if hasattr(self.image_encoder, "num_features"):
            self.num_features = self.image_encoder.num_features
        elif "resnet" in model_name:
            # Fallback for some models
            self.num_features = self.image_encoder.fc.in_features
            self.image_encoder.fc = torch.nn.Identity()
        elif "efficientvit" in model_name:
            self.num_features = self.image_encoder.head.classifier[-1].in_features
            self.image_encoder.head.classifier[-1] = torch.nn.Identity()

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

    def freeze_image_encoder(self) -> None:
        """Freeze all parameters in the image encoder."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def unfreeze_image_encoder(self) -> None:
        """Unfreeze all parameters in the image encoder."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True

    def freeze_early_backbone(self) -> None:
        """Freeze early backbone layers (conv1, bn1, layer1, layer2, layer3) in ResNet.

        This prevents the early feature extractors from adapting during
        continual learning, forcing adaptation to occur only in higher-level
        layers. Useful for preventing catastrophic forgetting of low-level
        features while allowing task-specific adaptation in deeper layers.

        Freezes:
        - conv1: Initial 7x7 convolution
        - bn1: Initial batch normalization
        - layer1: First residual block (Block 1)
        - layer2: Second residual block (Block 2)
        - layer3: Third residual block (Block 3)
        """
        layers_to_freeze = ["conv1", "bn1", "layer1", "layer2", "layer3"]
        frozen_params = 0
        frozen_tensors = 0

        for layer_name in layers_to_freeze:
            if hasattr(self.image_encoder, layer_name):
                layer = getattr(self.image_encoder, layer_name)
                for param in layer.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                    frozen_tensors += 1

        print(
            f"  Frozen early backbone: {frozen_params:,} parameters in {frozen_tensors} tensors (conv1, bn1, layer1, layer2, layer3)"
        )

    def freeze_all_except_lora(self) -> None:
        """Freeze all parameters except LoRA parameters.

        This is the key method for extreme parameter efficiency:
        - Freezes the entire network (encoder + head)
        - Only LoRA parameters remain trainable
        """
        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        # Then unfreeze only LoRA parameters
        for module in self.modules():
            if isinstance(module, LoRALinear | LoRAConv2d):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
                module.lora_B.requires_grad = True

    def apply_lora_to_head(self, rank: int = 4, alpha: float = 1.0) -> None:
        """Apply LoRA to the network head linear layers.

        This replaces standard Linear layers with LoRALinear layers,
        enabling parameter-efficient adaptation.

        Parameters
        ----------
        rank : int, optional
            Rank of LoRA decomposition. Lower = fewer parameters.
            Recommended: 1-8 for extreme efficiency, by default 4.
        alpha : float, optional
            LoRA scaling factor, by default 1.0.
        """
        # Replace Linear layers in network_head with LoRA versions
        replace_linear_with_lora(self.network_head, rank=rank, alpha=alpha)

        # Print LoRA configuration
        print_lora_info(self)

        # Print overall parameter efficiency
        stats = count_lora_parameters(self)
        print("\nOverall Model Parameter Efficiency:")
        print(f"  Trainable:  {stats['trainable']:,} ({stats['efficiency_pct']:.4f}%)")
        print(f"  Frozen:     {stats['frozen']:,}")
        print(f"  Total:      {stats['total']:,}\n")

    def apply_dual_lora(
        self,
        encoder_rank: int = 4,
        head_rank: int = 8,
        encoder_alpha: float = 16.0,
        head_alpha: float = 16.0,
    ) -> None:
        """Apply LoRA to both Image Encoder and Network Head with asymmetric bottlenecking.

        This is the recommended approach for handling long-term concept drift (seasonal shifts,
        camera degradation, etc.). The encoder gets a very low rank to prevent memorization of
        daily noise while allowing systematic seasonal adjustments. The head gets moderate rank
        for final mapping.

        Recommended Configuration:
        - Encoder: rank=2-4, alpha=16 (extreme bottleneck for seasonal drift only)
        - Head: rank=8-16, alpha=16 (moderate capacity for final regression)
        - Use asymmetric learning rates: encoder_lr=1e-5, head_lr=1e-4

        Parameters
        ----------
        encoder_rank : int, optional
            Rank for encoder LoRA. Very low (2-4) prevents memorization, by default 4.
        head_rank : int, optional
            Rank for head LoRA. Moderate (8-16) for final mapping, by default 8.
        encoder_alpha : float, optional
            Scaling factor for encoder LoRA, by default 16.0.
        head_alpha : float, optional
            Scaling factor for head LoRA, by default 16.0.
        """
        print("\n" + "=" * 80)
        print("Applying Dual-LoRA with Asymmetric Bottlenecking")
        print("=" * 80)
        print(f"Image Encoder: rank={encoder_rank}, alpha={encoder_alpha}")
        print(f"Network Head:  rank={head_rank}, alpha={head_alpha}")

        # Apply LoRA to image encoder with extreme low rank
        # For CNNs (ResNet), we need to replace Conv2d layers
        replace_linear_with_lora(self.image_encoder, rank=encoder_rank, alpha=encoder_alpha, replace_conv2d=True)

        # Apply LoRA to network head with moderate rank (only Linear layers)
        replace_linear_with_lora(self.network_head, rank=head_rank, alpha=head_alpha)

        # Print detailed LoRA configuration
        print("\nLoRA Layer Information:")
        print("-" * 80)
        print(f"{'Layer Name':<50} {'Type':<8} {'Rank':<6} {'Trainable':<12} {'Total':<12}")
        print("-" * 80)

        encoder_trainable = 0
        encoder_total = 0
        head_trainable = 0
        head_total = 0

        for name, layer in self.named_modules():
            if isinstance(layer, LoRALinear | LoRAConv2d):
                layer_type = "Linear" if isinstance(layer, LoRALinear) else "Conv2d"
                trainable = layer.num_trainable_parameters()
                total = layer.num_total_parameters()
                print(f"{name:<50} {layer_type:<8} {layer.rank:<6} {trainable:<12,} {total:<12,}")

                if "image_encoder" in name:
                    encoder_trainable += trainable
                    encoder_total += total
                elif "network_head" in name:
                    head_trainable += trainable
                    head_total += total

        print("-" * 80)
        print(f"{'Image Encoder LoRA':<50} {'':<6} {encoder_trainable:<12,} {encoder_total:<12,}")
        print(f"{'Network Head LoRA':<50} {'':<6} {head_trainable:<12,} {head_total:<12,}")
        print(f"{'TOTAL LoRA':<50} {'':<6} {encoder_trainable + head_trainable:<12,} {encoder_total + head_total:<12,}")

        # Print overall parameter efficiency
        stats = count_lora_parameters(self)
        print("\nOverall Model Parameter Efficiency:")
        print(f"  Trainable:  {stats['trainable']:,} ({stats['efficiency_pct']:.4f}%)")
        print(f"  Frozen:     {stats['frozen']:,}")
        print(f"  Total:      {stats['total']:,}")
        print("\nRecommended: Use asymmetric learning rates (encoder_lr=1e-5, head_lr=1e-4)")
        print("=" * 80 + "\n")

    def get_lora_param_groups(self, encoder_lr: float, head_lr: float, weight_decay: float = 0.0) -> list:
        """Get parameter groups for LoRA with separate learning rates.

        This allows asymmetric learning rates for encoder and head LoRA parameters,
        which is critical for stable dual-LoRA training.

        Parameters
        ----------
        encoder_lr : float
            Learning rate for encoder LoRA parameters (recommended: 1e-5).
        head_lr : float
            Learning rate for head LoRA parameters (recommended: 1e-4).
        weight_decay : float, optional
            Weight decay for both groups, by default 0.0.

        Returns
        -------
        param_groups : list
            List of parameter group dictionaries for optimizer.
        """
        encoder_lora_params = []
        head_lora_params = []

        for name, module in self.named_modules():
            if isinstance(module, LoRALinear | LoRAConv2d):
                if "image_encoder" in name:
                    encoder_lora_params.extend([module.lora_A, module.lora_B])
                elif "network_head" in name:
                    head_lora_params.extend([module.lora_A, module.lora_B])

        param_groups = []
        if len(encoder_lora_params) > 0:
            param_groups.append(
                {"params": encoder_lora_params, "lr": encoder_lr, "weight_decay": weight_decay, "name": "encoder_lora"}
            )
        if len(head_lora_params) > 0:
            param_groups.append({"params": head_lora_params, "lr": head_lr, "weight_decay": weight_decay, "name": "head_lora"})

        return param_groups

    def merge_lora_weights(self) -> None:
        """Merge LoRA weights into base model for inference efficiency.

        After merging, LoRA layers behave like standard layers,
        eliminating the overhead of low-rank computation.
        """
        for module in self.modules():
            if isinstance(module, LoRALinear | LoRAConv2d):
                module.merge_weights()
                module.merge_weights()

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
        elif loss_name == "Berhu":
            return MeanAdaptiveBerHuLoss()
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
            mean = F.softplus(mean)
            # Softplus + epsilon ensures variance > 0
            log_var = out[:, 1].unsqueeze(1)
            variance = F.softplus(log_var) + 1e-6
            return mean, variance
        else:
            # Standard Regression: [Batch, 1]
            out = F.softplus(out)
            return out, None

    def extract_proto_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract prototype features from the deep attention stage.

        For EfficientViT models (those with ``image_encoder.stages``), runs the
        input through the full encoder with a forward hook on ``stages[3]`` (the
        linear attention block) and returns the global-average-pooled output
        ``[B, proto_dim]``.  Early CNN stages (``stages[0]`` / ``stages[1]``)
        learn site-local texture and horizon lines and are deliberately excluded
        from the prototype representation — their output is never compared
        across clients.

        For other architectures (e.g. ResNet) falls back to the full encoder
        output so that legacy checkpoints work unchanged.

        Returns
        -------
        proto_feats : Tensor, shape (B, proto_dim)
            Stage-3 GAP features (EfficientViT) or full encoder output (ResNet).
        """
        if hasattr(self.image_encoder, "stages"):
            _out: dict[str, torch.Tensor] = {}
            _handle = self.image_encoder.stages[3].register_forward_hook(lambda _m, _i, o: _out.update({"feat": o}))
            try:
                self.image_encoder(x)
            finally:
                _handle.remove()
            feat = _out["feat"]  # [B, C, H, W]
            return feat.mean(dim=(-2, -1))  # GAP → [B, proto_dim]
        else:
            return self.image_encoder(x)

    def forward_with_features(
        self, x: torch.Tensor, irradiance_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Forward pass that also returns stage-3 prototype features.

        Used by FedProto for prototype regularisation.  A single encoder
        forward pass is performed; prototype features are captured via a
        lightweight hook on ``stages[3]`` and must **not** be confused with
        the full encoder output used for the irradiance prediction.

        Architecture contract (EfficientViT B2)
        -----------------------------------------
        * **Private stem** — ``stages[0]`` / ``stages[1]`` (early CNN blocks)
          learn site-local textures and horizon geometry.  Their gradients are
          untouched by the prototype loss and weights stay fully local.
        * **Shared core** — ``stages[3]`` (linear attention) produces
          disentangled cloud-physics representations suitable for cross-client
          prototype comparison.  The prototype MSE loss applies here.
        * **Private head** — ``network_head`` adapts to the local pyranometer
          calibration offset without any federated penalty.

        For ResNet checkpoints the full encoder output is returned unchanged
        (backward-compatible fallback).

        Returns
        -------
        mean : Tensor
            Predicted irradiance (after softplus).
        variance : Tensor or None
            Predicted variance (probabilistic mode) or None.
        proto_feats : Tensor, shape (B, proto_dim)
            Stage-3 GAP features (EfficientViT) or full encoder output (ResNet).
        """
        # Capture stage-3 spatial output during the single encoder forward pass.
        _proto_holder: dict[str, torch.Tensor] = {}
        if hasattr(self.image_encoder, "stages"):
            _handle = self.image_encoder.stages[3].register_forward_hook(lambda _m, _i, o: _proto_holder.update({"feat": o}))

        try:
            img_feats = self.image_encoder(x)  # [B, encoder_dim] — used for prediction
        finally:
            if hasattr(self.image_encoder, "stages"):
                _handle.remove()

        proto_feats = _proto_holder["feat"].mean(dim=(-2, -1)) if "feat" in _proto_holder else img_feats

        fused = torch.cat([img_feats, irradiance_history], dim=1)
        out = self.network_head(fused)

        if self.is_probabilistic:
            mean = F.softplus(out[:, 0].unsqueeze(1))
            variance = F.softplus(out[:, 1].unsqueeze(1)) + 1e-6
            return mean, variance, proto_feats
        else:
            out = F.softplus(out)
            return out, None, proto_feats

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
        elif self.loss_name == "Berhu":
            loss = self.loss_fn(pred_mean * 1000, target_irradiances * 1000)  # Scale to W/m^2 for BerHu loss
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
