from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd

from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics


@click.command()
@click.option(
    "--periods-path",
    help="Data frame with evaluation periods",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default="data/prepared/periods.pickle",
)
@click.option(
    "--clear-sky-path",
    help="Path to clear sky irradiance data",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default="data/prepared/clear_sky_ineichen.csv",
)
@click.option(
    "--forecasting-horizon",
    help="Forecasting horizon in minutes",
    type=int,
    default=15,
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    help="Number of days to evaluate. -1 means all days.",
)
@click.option(
    "--start-day",
    type=str,
    default="0101",
    help="Starting day in MMDD format (e.g., 0101)",
)
@click.option("--save-results", is_flag=True, help="Whether to save the predictions and results to files")
def evaluate_smart_persistence_model(
    periods_path: Path, clear_sky_path: Path, forecasting_horizon: int, num_days: int, start_day: str, save_results: bool
):
    # Extract dataset name from periods path
    dataset_name = periods_path.parent.name.split("_")[0]

    # Load clear sky data
    clear_sky_df = pd.read_csv(clear_sky_path)
    clear_sky_df["datetime"] = pd.to_datetime(clear_sky_df["datetime"], utc=False)
    clear_sky_dict = dict(zip(clear_sky_df["datetime"], clear_sky_df["ghi_clear"], strict=True))

    with periods_path.open("rb") as f:
        test_periods = pd.read_pickle(f)

    if "folsom" in str(periods_path).lower():
        # Use only 2016 data for Folsom
        test_periods = list(filter(lambda p: p["history"][-1]["image_name"].startswith("2016"), test_periods))

    # Group indices by day and filter if needed
    day_index_map = group_indices_by_day(test_periods)
    print(f"Total samples: {len(test_periods)}")
    print(f"Number of unique days: {len(day_index_map)}\n")

    # Determine which days to evaluate
    selected_indices = None
    if num_days > 0:
        all_days = generate_year_days()
        if start_day not in all_days:
            raise ValueError(f"Invalid start_day '{start_day}'. Must be in MMDD format (0101-1231)")
        start_idx = all_days.index(start_day)
        selected_days = all_days[start_idx : start_idx + num_days]

        # Collect indices for selected days
        selected_indices = []
        for day_key in selected_days:
            if day_key in day_index_map:
                selected_indices.extend(day_index_map[day_key])

        # Filter test_periods to only include selected indices
        test_periods = [test_periods[idx] for idx in selected_indices]

        print(f"Evaluating {len(selected_days)} days starting from {start_day}")
        print(f"Total samples after filtering: {len(test_periods)}\n")
    else:
        print("Evaluating all days\n")

    predictions_data = []
    day_predictions = defaultdict(list)
    day_targets = defaultdict(list)
    current_day = None
    current_day_preds = []
    current_day_targets = []
    naive_count = 0

    print("Day    MAE        RMSE       MAPE       WAPE       SMAPE      Samples ")
    print("------------------------------------------------------------------------")

    for p in test_periods:
        # Get the last observed irradiance and its clear sky value
        last_history = p["history"][-1]
        last_irradiance = last_history["irradiance"]
        # Extract timestamp from image_name (format: yyyymmdd_HHMMSS.ext)
        image_name = last_history["image_name"]
        date_time_str = Path(image_name).stem
        last_timestamp = pd.to_datetime(date_time_str, format="%Y%m%d_%H%M%S", utc=False)

        # Get clear sky irradiance for last observation
        last_clear_sky = clear_sky_dict.get(last_timestamp)

        # Get clear sky irradiance for target time
        target_timestamp = last_timestamp + pd.Timedelta(minutes=forecasting_horizon)
        target_clear_sky = clear_sky_dict.get(target_timestamp)

        if last_clear_sky is not None and target_clear_sky is not None and last_clear_sky > 0:
            # Calculate clear sky index from last observation
            kc_last = last_irradiance / last_clear_sky
            kc_last = min(max(kc_last, 0.0), 2.0)  # clamp to [0, 2]

            # Predict target irradiance using smart persistence
            pred_irradiance = kc_last * target_clear_sky
        else:
            # Fallback to naive persistence if clear sky data is missing
            pred_irradiance = last_irradiance
            naive_count += 1

        target_irradiance = p["target_irradiance"]

        # Extract day key (MMDD format)
        day_key = image_name[4:8] if len(image_name) >= 8 else None

        # Store prediction data
        predictions_data.append(
            {
                "timestamp": image_name,
                "day": day_key,
                "prediction": pred_irradiance,
                "target": target_irradiance,
                "error": pred_irradiance - target_irradiance,
                "absolute_error": abs(pred_irradiance - target_irradiance),
            }
        )

        # Accumulate per-day predictions and targets
        if day_key is not None:
            day_predictions[day_key].append(pred_irradiance)
            day_targets[day_key].append(target_irradiance)

            # Initialize current day tracking
            if current_day is None:
                current_day = day_key

            # Print metrics when day changes
            if day_key != current_day:
                day_metrics = compute_metrics(
                    np.array(current_day_preds),
                    np.array(current_day_targets),
                )
                print(
                    f"{current_day:<6} {day_metrics['mae']:<10.4f} "
                    f"{day_metrics['rmse']:<10.4f} {day_metrics['mape']:<16.2%} "
                    f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% "
                    f"{day_metrics['count']:<8}"
                )
                current_day = day_key
                current_day_preds = []
                current_day_targets = []

            current_day_preds.append(pred_irradiance)
            current_day_targets.append(target_irradiance)

    # Print metrics for the last day
    if current_day is not None and current_day_preds:
        day_metrics = compute_metrics(
            np.array(current_day_preds),
            np.array(current_day_targets),
        )
        print(
            f"{current_day:<6} {day_metrics['mae']:<10.4f} "
            f"{day_metrics['rmse']:<10.4f} {day_metrics['mape']:<16.2%} "
            f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% "
            f"{day_metrics['count']:<8}"
        )

    print(f"\nUsed naive persistence for {naive_count} out of {len(test_periods)} periods.")

    # Compute overall metrics
    all_preds = np.array([p["prediction"] for p in predictions_data])
    all_targets = np.array([p["target"] for p in predictions_data])
    overall_metrics = compute_metrics(all_preds, all_targets)

    # Compute per-day metrics
    per_day_metrics = {}
    for day_key in sorted(day_predictions.keys()):
        day_preds = np.array(day_predictions[day_key])
        day_targets_arr = np.array(day_targets[day_key])
        per_day_metrics[day_key] = compute_metrics(day_preds, day_targets_arr)

    # Print overall results
    print(f"\n{'='*60}")
    print(f"Smart Persistence Model Evaluation: {dataset_name}")
    print(f"{'='*60}\n")
    print(f"Forecasting horizon: {forecasting_horizon} minutes")
    print("Overall Metrics:")
    print(f"  MAE:   {overall_metrics['mae']:.4f}")
    print(f"  RMSE:  {overall_metrics['rmse']:.4f}")
    print(f"  MAPE:  {overall_metrics['mape']:.2%}")
    print(f"  WAPE:  {overall_metrics['wape']:.2f}%")
    print(f"  SMAPE: {overall_metrics['smape']:.2f}%")
    print(f"  Samples: {overall_metrics['count']}")

    # Save results if requested
    if save_results:
        results_file = f"{dataset_name}_smart_persistence_{forecasting_horizon}min_results.txt"
        predictions_file = f"{dataset_name}_smart_persistence_{forecasting_horizon}min_predictions.csv"

        # Save predictions to CSV
        df_predictions = pd.DataFrame(predictions_data)
        df_predictions.to_csv(predictions_file, index=False)
        print(f"\nPredictions saved to {predictions_file}")

        # Save results to text file
        with Path(results_file).open("w") as f:
            f.write(f"Smart Persistence Model Evaluation: {dataset_name}\n")
            f.write(f"Periods path: {periods_path}\n")
            f.write(f"Clear sky path: {clear_sky_path}\n")
            f.write(f"Forecasting horizon: {forecasting_horizon} minutes\n")
            f.write(f"Naive persistence fallbacks: {naive_count}/{len(test_periods)}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Overall Metrics:\n")
            f.write(f"  MAE:   {overall_metrics['mae']:.4f}\n")
            f.write(f"  RMSE:  {overall_metrics['rmse']:.4f}\n")
            f.write(f"  MAPE:  {overall_metrics['mape']:.4f}%\n")
            f.write(f"  WAPE:  {overall_metrics['wape']:.2f}%\n")
            f.write(f"  SMAPE: {overall_metrics['smape']:.2f}%\n")
            f.write(f"  Samples: {overall_metrics['count']}\n\n")

            f.write("Per-Day Metrics:\n")
            f.write(f"{'Day':<6} {'MAE':<10} {'RMSE':<10} {'MAPE':<16} {'WAPE':<10} {'SMAPE':<10} {'Samples':<8}\n")
            f.write("-" * 78 + "\n")
            for day_key, metrics in per_day_metrics.items():
                f.write(
                    f"{day_key:<6} {metrics['mae']:<10.4f} "
                    f"{metrics['rmse']:<10.4f} {metrics['mape']:<16.2%} "
                    f"{metrics['wape']:<10.2f}% {metrics['smape']:<10.2f}% "
                    f"{metrics['count']:<8}\n"
                )

        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    evaluate_smart_persistence_model()
