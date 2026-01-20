from pathlib import Path
from time import perf_counter

import click
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cirr.models.forecaster import Forecaster


@click.command()
@click.option("--checkpoint-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--iterations", type=int, default=1000)
def main(checkpoint_path: Path, iterations: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("###################################")
    print(f"Device: {device}")
    print("###################################")
    img_tensor = torch.randn(1, 3, 128, 128).to(device)
    irradiance_history_tensor = torch.randn(1, 4).to(device)
    target_irradiance_tensor = torch.randn(1, 1).to(device)

    # Inference
    model = Forecaster.load_from_checkpoint(checkpoint_path)
    model.eval()

    # warm-up model
    for _ in range(10):
        _ = model(img_tensor, irradiance_history_tensor)

    inference_times = []
    for _ in tqdm(range(iterations)):
        inference_start = perf_counter()
        _ = model(img_tensor, irradiance_history_tensor)
        inference_time = perf_counter() - inference_start
        inference_times.append(inference_time)
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    print("###################################")
    print(f"Average inference time: {avg_inference_time*1000:.4f} ms")
    print(f"STD inference time: {std_inference_time*1000:.4f} ms")
    print("###################################")

    # Full graph train
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model = Forecaster.load_from_checkpoint(checkpoint_path)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    training_times = []
    for _ in tqdm(range(iterations)):
        training_start = perf_counter()
        model_output = model(img_tensor, irradiance_history_tensor)
        loss = criterion(model_output, target_irradiance_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        training_time = perf_counter() - training_start
        training_times.append(training_time)
    avg_training_time = np.mean(training_times)
    std_training_time = np.std(training_times)

    print("###################################")
    print(f"Average full graph training time: {avg_training_time*1000:.4f} ms")
    print(f"STD full graph training time: {std_training_time*1000:.4f} ms")
    print("###################################")

    # Head layers fine-tuning
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model = Forecaster.load_from_checkpoint(checkpoint_path)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for param in model.named_parameters():
        if "image_encoder" in param[0]:
            param[1].requires_grad = False

    head_fine_tuning_times = []
    for _ in tqdm(range(iterations)):
        head_fine_tuning_start = perf_counter()
        model_output = model(img_tensor, irradiance_history_tensor)
        loss = criterion(model_output, target_irradiance_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        head_fine_tuning_time = perf_counter() - head_fine_tuning_start
        head_fine_tuning_times.append(head_fine_tuning_time)
    avg_head_fine_tuning_time = np.mean(head_fine_tuning_times)
    std_head_fine_tuning_time = np.std(head_fine_tuning_times)

    print("###################################")
    print(f"Average head layers fine-tuning time: {avg_head_fine_tuning_time*1000:.4f} ms")
    print(f"STD head layers fine-tuning time: {std_head_fine_tuning_time*1000:.4f} ms")
    print("###################################")


if __name__ == "__main__":
    main()
