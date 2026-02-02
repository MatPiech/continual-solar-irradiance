from pathlib import Path
import pickle

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option("--cleaned-dataframe-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-path", type=click.Path(path_type=Path), required=True)
@click.option("--irradiance-col", type=str, default="ghi")
@click.option("--history-size", type=int, default=4)
@click.option("--time-shift", type=int, default=5)
@click.option("--time-window", type=int, default=15)
@click.option("--start-date", type=str, default=None, help="Start date for target time range. Format: 'YYYY-MM-DD HH:MM:SS'")
@click.option("--end-date", type=str, default=None, help="End date for target time range. Format: 'YYYY-MM-DD HH:MM:SS'")
def export_periods(
    cleaned_dataframe_path: Path,
    output_path: Path,
    irradiance_col: str,
    history_size: int,
    time_shift: int,
    time_window: int,
    start_date: str,
    end_date: str,
):
    periods = []

    df = pd.read_csv(
        cleaned_dataframe_path,
        parse_dates=["datetime"],
        index_col="datetime",
    )
    # Parse the datetime index from strings (includes timezone info)
    df.index = pd.to_datetime(df.index, utc=True)

    # Parse time range if provided
    start_timestamp = pd.Timestamp(start_date) if start_date else None
    end_timestamp = pd.Timestamp(end_date) if end_date else None

    if start_timestamp or end_timestamp:
        range_info = "Filtering target timestamps: "
        if start_timestamp:
            range_info += f"from {start_timestamp} "
        if end_timestamp:
            range_info += f"to {end_timestamp}"
        print(range_info)

    print(f"Min irradiance: {df[irradiance_col].min()}")
    print(f"Max irradiance: {df[irradiance_col].max()}")
    print(f"Mean irradiance: {df[irradiance_col].mean()}")
    print(f"Irradiance std: {df[irradiance_col].std()}")

    for t, _ in tqdm(df.iterrows(), total=len(df)):
        t = pd.Timestamp(t)
        t_history = [t - pd.Timedelta(minutes=time_shift * i) for i in range(history_size - 1, 0, -1)]
        t_target = t + pd.Timedelta(minutes=time_window)

        # Check if target timestamp is within the specified range
        if start_timestamp and t_target < start_timestamp:
            continue
        if end_timestamp and t_target > end_timestamp:
            continue

        if all(x in df.index for x in [*t_history, t_target]):
            target_irradiance = df.loc[t_target][irradiance_col]
            if isinstance(target_irradiance, pd.Series):
                target_irradiance = target_irradiance.iloc[0]
            if target_irradiance >= 0.0:
                history_items = []
                for _t in sorted([*t_history, t]):
                    img_name = df.loc[_t]["image_name"]
                    if isinstance(img_name, pd.Series):
                        img_name = img_name.iloc[0]
                    irr = df.loc[_t][irradiance_col]
                    if isinstance(irr, pd.Series):
                        irr = irr.iloc[0]
                    history_items.append({"image_name": img_name, "irradiance": irr})

                periods.append(
                    {
                        "history": history_items,
                        "target_irradiance": target_irradiance,
                    }
                )

    print(f"Number of periods: {len(periods)}")
    with output_path.open("wb") as f:
        pickle.dump(periods, f)


if __name__ == "__main__":
    export_periods()
