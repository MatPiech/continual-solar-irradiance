import copy

import numpy as np
import torch

from cirr.models.forecaster import Forecaster


class Client:
    """Client that decides locally whether to train on each incoming sample."""

    def __init__(self, client_id: str, model_params, device: str = "cuda"):
        self.id = client_id
        self.device = device

        # Initialize local model and optimizer once per client
        self.model = Forecaster(
            model_name="timm-resnet50",
            pretrained=True,
            input_channels=3,
            loss_function="GNLL",
            lr=0.001,
            lr_patience=5,
        ).to(self.device)
        self.model.load_state_dict(model_params)
        self.optimizer = self.model.configure_optimizers()["optimizer"]

        # Track local uncertainty history for dynamic thresholding
        self.uncertainty_history: list[float] = []
        self.history_window = 100
        self.warmup_samples = 10

    def process_day(self, global_weights, day_loader, threshold_percentile: float = 75.0):
        """Run inference sample-by-sample for a day; train on high-uncertainty samples."""
        # Store initial weights to compute change
        initial_weights = copy.deepcopy(global_weights)
        self.model.load_state_dict(global_weights)
        self.model.eval()

        triggered = False
        has_data = False
        samples_processed = 0
        samples_trained = 0

        for batch in day_loader:
            has_data = True
            samples_processed += batch[0].size(0)  # batch size
            batch = self._move_to_device(batch)

            with torch.no_grad():
                _, variance = self.model(batch[0], batch[1])
                current_uncertainty = variance.mean().item()

            self._update_uncertainty_history(current_uncertainty)
            threshold = self._compute_threshold(threshold_percentile)

            if current_uncertainty > threshold:
                triggered = True
                samples_trained += batch[0].size(0)
                self._train_on_batch(batch)

        # Compute weight change (L2 norm of difference)
        final_weights = self.model.state_dict()
        weight_change_mb = self._compute_weight_change(initial_weights, final_weights)

        # Always return weights for federated averaging, even if no data or no training
        return (
            copy.deepcopy(final_weights),
            triggered and has_data,
            samples_processed,
            samples_trained,
            weight_change_mb,
        )

    def _train_on_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model.training_step(batch, 0)
        loss.backward()
        self.optimizer.step()

    def _update_uncertainty_history(self, uncertainty: float):
        self.uncertainty_history.append(uncertainty)
        if len(self.uncertainty_history) > self.history_window:
            self.uncertainty_history.pop(0)

    def _compute_threshold(self, percentile: float) -> float:
        if len(self.uncertainty_history) < self.warmup_samples:
            return 0.0
        return float(np.percentile(self.uncertainty_history, percentile))

    def _compute_weight_change(self, weights1, weights2):
        """Compute L2 norm of weight difference in MB."""
        total_change = 0.0
        for key in weights1:
            if weights1[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                diff = weights1[key].float() - weights2[key].float()
                total_change += torch.sum(diff**2).item()
        total_change**0.5
        # Approximate MB (assuming float32 = 4 bytes per parameter)
        num_params = sum(p.numel() for p in weights2.values())
        size_mb = (num_params * 4) / (1024**2)
        return size_mb

    def _move_to_device(self, batch):
        # Keep metadata (image names) on CPU; move tensors only.
        return [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
