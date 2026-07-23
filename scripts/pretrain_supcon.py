"""SupCon pretraining for the vision encoder on Folsom Years 1-2."""

from __future__ import annotations

from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cirr.data.loading import load_dataset
from cirr.data.transforms import input_channels
from cirr.losses import SupConLoss
from cirr.models.forecaster import Forecaster
from cirr.physics.csi import compute_csi, discretize_csi


def _collate(batch):
    images = torch.stack([b[0] for b in batch])
    ghi_clear = torch.stack([b[2] for b in batch])
    targets = torch.stack([b[3] for b in batch]).view(-1)
    labels = []
    for t, cs in zip(targets, ghi_clear, strict=True):
        csi = compute_csi(t.item(), cs.item())
        labels.append(discretize_csi(csi))
    return images, torch.tensor(labels, dtype=torch.long)


@click.command()
@click.option("--data-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--epochs", type=int, default=5, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("checkpoints/supcon_encoder.pt"))
@click.option("--device", type=str, default="cuda")
def main(data_root: Path, epochs: int, batch_size: int, lr: float, output: Path, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dataset = load_dataset(data_root, split="training", batch_size=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate)

    model = Forecaster(
        model_name="timm-resnet50",
        pretrained=True,
        input_channels=input_channels(optical_flow="none"),
        loss_function="MSE",
        lr=lr,
        lr_patience=2,
    ).to(device)

    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.image_encoder.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.image_encoder.parameters(), lr=lr)
    criterion = SupConLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for images, labels in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            feats = model.extract_proto_features(images)
            loss = criterion(feats, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch + 1}: supcon_loss={total_loss / max(n_batches, 1):.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_encoder": model.image_encoder.state_dict()}, output)
    print(f"Saved encoder checkpoint to {output}")


if __name__ == "__main__":
    main()
