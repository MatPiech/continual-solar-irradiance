from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from typing import Any

import click
import torch
from torch.utils.data import DataLoader, Subset

from cirr.datamodules.forecasting import ForecastingDataModule
from cirr.fl.client import Client
from cirr.fl.server import federated_averaging
from cirr.models.forecaster import Forecaster

SITES = {
    "folsom": Path("data/folsom_prepared"),
    "karpicko": Path("data/karpicko_prepared"),
    "poznan": Path("data/poznan_prepared"),
}


def generate_year_days() -> list[str]:
    """Generate all days of a standard year (365 days) as MMDD strings."""
    days = []
    start_date = datetime(2016, 1, 1)
    for day_offset in range(366):  # 2016 is a leap year
        current_date = start_date + timedelta(days=day_offset)
        # Skip Feb 29 to make it a standard 365-day year
        if current_date.month == 2 and current_date.day == 29:
            continue
        days.append(current_date.strftime("%m%d"))  # MMDD format
    return days


def group_indices_by_day(periods: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Split periods into daily slices using MMDD format (year-invariant)."""
    day_to_indices = defaultdict(list)
    sorted_indices = sorted(range(len(periods)), key=lambda idx: periods[idx]["history"][-1]["image_name"])
    for idx in sorted_indices:
        image_name = periods[idx]["history"][-1]["image_name"]
        # Extract MMDD only (positions 4-8 of YYYYMMDD), year-invariant
        day_key = image_name[4:8]  # MMDD
        day_to_indices[day_key].append(idx)
    return day_to_indices


def setup_global_model(checkpoint_path: str) -> dict[str, Any]:
    """Initialize global model and load checkpoint weights."""
    global_model = Forecaster(
        model_name="timm-resnet50",
        pretrained=True,
        input_channels=3,
        loss_function="GNLL",
        lr=0.001,
        lr_patience=5,
    ).to("cuda")
    global_model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return global_model.state_dict()


def load_site_data(site_name: str, data_path: Path) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Load test dataset and build day index map for a site."""
    datamodule = ForecastingDataModule(
        root_data_path=data_path,
        periods_path=data_path / "periods.pickle",
        augment=False,
        image_size=[128, 128],
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        batch_size=1,
        workers=4,
        seed=42,
    )
    datamodule.setup(None)
    test_dataset = datamodule._test_dataset
    day_index_map = group_indices_by_day(test_dataset._periods)
    print(f"Loaded {site_name}: {len(test_dataset._periods)} periods")
    return test_dataset, day_index_map


def create_clients(client_ids: list[str], global_weights: dict[str, Any], num_clients: int) -> list[Client]:
    """Create federated learning clients."""
    clients = []
    for i in range(num_clients):
        client_id = client_ids[i] if i < len(client_ids) else f"client_{i}"
        clients.append(Client(client_id=client_id, model_params=global_weights))
    return clients


def run_fl_round(
    day_key: str,
    clients: list[Client],
    global_weights: dict[str, Any],
    site_data: list[tuple[Any, dict[str, list[int]]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one federated learning round for a day."""
    client_updates = []
    trained_count = 0
    has_data_any_site = False
    client_stats = []

    for client_idx, client in enumerate(clients):
        # Cycle through sites
        site_idx = client_idx % len(site_data)
        test_dataset, day_index_map = site_data[site_idx]
        day_indices = day_index_map.get(day_key, [])
        has_data = len(day_indices) > 0

        if has_data:
            has_data_any_site = True
            day_loader = DataLoader(
                Subset(test_dataset, day_indices),
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )
        else:
            # Empty loader for days without data
            day_loader = DataLoader(
                Subset(test_dataset, []),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )

        update, trained, samples_processed, samples_trained, weight_change = client.process_day(global_weights, day_loader)
        client_updates.append(update)
        if trained:
            trained_count += 1

        client_stats.append(
            {
                "client_id": client.id,
                "samples_processed": samples_processed,
                "samples_trained": samples_trained,
                "weight_change_mb": weight_change,
                "trained": trained,
            }
        )

    new_global_weights, global_weight_change = federated_averaging(global_weights, client_updates)

    round_stats = {
        "trained_count": trained_count,
        "has_data": has_data_any_site,
        "client_stats": client_stats,
        "global_weight_change_mb": global_weight_change,
    }

    return new_global_weights, round_stats


def run_simulation(
    checkpoint_path: Path,
    sites: list[tuple[str, Path]],
    num_rounds: int,
    start_day: str,
) -> dict[str, Any]:
    """Run federated learning simulation across multiple sites.

    Parameters
    ----------
    checkpoint_path : Path
        Path to initial model checkpoint, by default "checkpoints/epoch_007.ckpt".
    sites : list[tuple[str, Path]]
        List of (site_name, data_path) tuples. Defaults to folsom, karpicko, poznan.
    num_rounds : int
        Number of rounds (days) to simulate. None means all 365 days, by default None.
    start_day : str
        Starting day in MMDD format (e.g., '0101'). None means start from 0101, by default None.

    Returns
    -------
    dict[str, Any]
        Training history dictionary with rounds, bandwidth savings, and transfer costs.
    """
    num_clients = len(sites)

    # Setup
    print("Initializing federated learning simulation...")
    global_weights = setup_global_model(checkpoint_path)

    # Load data from all sites
    site_data = []
    for site_name, data_path in sites:
        if data_path.exists():
            test_dataset, day_index_map = load_site_data(site_name, data_path)
            site_data.append((test_dataset, day_index_map))
        else:
            print(f"Warning: Site data not found at {data_path}")

    if not site_data:
        raise ValueError("No valid site data found")

    # Create clients
    client_ids = [site_name for site_name, _ in sites]
    clients = create_clients(client_ids, global_weights, num_clients)
    print(f"Created {len(clients)} clients: {[c.id for c in clients]}\n")

    # Generate training days
    all_days = generate_year_days()

    # Apply start_day and num_rounds filters
    if start_day:
        if start_day not in all_days:
            raise ValueError(f"Invalid start_day '{start_day}'. Must be in MMDD format (0101-1231)")
        start_idx = all_days.index(start_day)
        all_days = all_days[start_idx:]

    if num_rounds > 0:
        all_days = all_days[:num_rounds]

    print(f"Simulating {len(all_days)} rounds starting from day {all_days[0]}\n")

    # History tracking
    history = {
        "rounds": [],
        "bandwidth_savings": [],
        "has_data": [],
        "total_samples_processed": [],
        "total_samples_trained": [],
        "global_weight_changes": [],
        "upload_costs": [],
        "download_costs": [],
        "total_transfer_costs": [],
    }

    # Run federated learning rounds
    print("Starting federated learning rounds...\n")
    for round_idx, day_key in enumerate(all_days):
        global_weights, round_stats = run_fl_round(day_key, clients, global_weights, site_data)

        # Aggregate statistics
        total_samples_processed = sum(c["samples_processed"] for c in round_stats["client_stats"])
        total_samples_trained = sum(c["samples_trained"] for c in round_stats["client_stats"])

        savings = ((len(clients) - round_stats["trained_count"]) / len(clients)) * 100

        # Print round summary
        print(f"Round {round_idx:3d} | Day {day_key}")
        print(f"  Clients trained: {round_stats['trained_count']}/{len(clients)} | Bandwidth savings: {savings:.1f}%")
        print(f"  Total samples: {total_samples_processed} processed, {total_samples_trained} trained")

        # Print per-client statistics
        for client_stat in round_stats["client_stats"]:
            trained_marker = "✓" if client_stat["trained"] else "✗"
            print(
                f"    {trained_marker} {client_stat['client_id']:8s}: "
                f"{client_stat['samples_processed']:4d} samples "
                f"({client_stat['samples_trained']:4d} trained), "
                f"weight Δ: {client_stat['weight_change_mb']:.2f} MB"
            )

        # Print aggregation statistics
        print(f"  Global weight Δ: {round_stats['global_weight_change_mb']:.2f} MB")

        # Update history
        history["rounds"].append(round_idx)
        history["bandwidth_savings"].append(savings)
        history["has_data"].append(round_stats["has_data"])
        history["total_samples_processed"].append(total_samples_processed)
        history["total_samples_trained"].append(total_samples_trained)
        history["global_weight_changes"].append(round_stats["global_weight_change_mb"])

    print(f"\nSimulation complete. Total rounds: {len(all_days)}")
    print(f"Total data transfer: {sum(history['total_transfer_costs']):.2f} MB")
    print(f"Average per round: {sum(history['total_transfer_costs']) / len(all_days):.2f} MB")
    return history


@click.command()
@click.option(
    "--checkpoint",
    default="checkpoints/epoch_007.ckpt",
    help="Path to initial model checkpoint",
)
@click.option(
    "--sites-name",
    type=str,
    multiple=True,
    default=None,
    help="List of site_names. If not provided, defaults to folsom, karpicko, poznan.",
)
@click.option(
    "--num-rounds",
    type=int,
    default=-1,
    help="Number of rounds (days) to simulate. Default: all 365 days",
)
@click.option(
    "--start-day",
    type=str,
    default="0101",
    help="Starting day in MMDD format (e.g., 0101 for January 1). Default: 0101",
)
@click.option(
    "--save-history",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save simulation history as pickle file",
)
def main(checkpoint: str, sites_name: list[str] | None, num_rounds: int, start_day: str, save_history: Path | None):
    """Run federated learning simulation for solar irradiance forecasting."""
    if sites_name is None:
        sites = []
        for site in sites_name:
            sites.append((site, SITES[site]))
    else:
        sites = [
            ("folsom", Path("data/folsom_prepared")),
            ("karpicko", Path("data/karpicko_prepared")),
            ("poznan", Path("data/poznan_prepared")),
        ]

    history = run_simulation(
        checkpoint_path=checkpoint,
        num_rounds=num_rounds,
        start_day=start_day,
        sites=sites,
    )
    print("\nSimulation completed successfully!")
    print(f"Total rounds: {len(history['rounds'])}")

    if save_history is not None:
        with save_history.open("wb") as f:
            pickle.dump(history, f)
        print(f"Simulation history saved to {save_history}")


if __name__ == "__main__":
    main()
