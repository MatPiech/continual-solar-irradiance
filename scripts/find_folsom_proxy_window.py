import click
import pandas as pd


def find_optimal_proxy_window(df, window_size=45, ghi_col="ghi", clear_ghi_col="ghi_clear"):
    """
    Identifies the most volatile N-day window in an irradiance dataset
    based on the variance of the Clear Sky Index (k_c).

    Parameters:
    - df: pandas DataFrame with a DatetimeIndex (minute frequency)
    - window_size: Int, number of days for the proxy dataset (default 45)
    - ghi_col: String, column name for measured Global Horizontal Irradiance
    - clear_ghi_col: String, column name for Clear Sky GHI

    Returns:
    - start_date, end_date: The dates for the optimal proxy window
    - rolling_variance: The full pandas Series of the rolling variance
    """

    # 1. Filter out night and low-sun angles to avoid division by zero
    # A threshold of 20-50 W/m^2 is standard in literature to exclude dawn/dusk noise
    daylight_mask = df[clear_ghi_col] > 20.0
    df_day = df[daylight_mask].copy()

    # 2. Calculate the Clear Sky Index (k_c)
    df_day["k_c"] = df_day[ghi_col] / df_day[clear_ghi_col]

    # Optional but recommended: Clip extreme outliers caused by cloud enhancement
    # (where measured GHI temporarily exceeds clear sky due to reflection)
    df_day["k_c"] = df_day["k_c"].clip(lower=0, upper=1.5)

    # 3. Calculate daily intra-day variance of k_c
    # Resample by Day ('D') and compute statistical variance
    daily_var = df_day["k_c"].resample("D").var()

    # Fill any NaN days (e.g., missing data) with 0 so the rolling sum doesn't break
    daily_var = daily_var.fillna(0)

    # 4. Calculate the rolling N-day sum of daily variances
    # Note: min_periods=window_size ensures we only consider full 45-day blocks
    rolling_var = daily_var.rolling(window=window_size, min_periods=window_size).sum()

    # 5. Identify the window with the maximum cumulative variance
    # idxmax() returns the END date of the rolling window
    optimal_end_date = rolling_var.idxmax()

    # Calculate the exact start date of this window
    optimal_start_date = optimal_end_date - pd.Timedelta(days=window_size - 1)

    print(f"Optimal {window_size}-Day Proxy Window Found:")
    print(f"Start Date: {optimal_start_date.strftime('%Y-%m-%d')}")
    print(f"End Date:   {optimal_end_date.strftime('%Y-%m-%d')}")
    print(f"Max Cumulative k_c Variance: {rolling_var.max():.4f}")

    return optimal_start_date, optimal_end_date, rolling_var


def normalize_mixed_timezone(series, local_tz):
    raw = series.astype(str)
    tz_mask = raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$")

    parsed = pd.to_datetime(raw, format="%Y%m%d_%H%M%S", errors="coerce")
    if (~tz_mask).any():
        remaining = ~tz_mask & parsed.isna()
        if remaining.any():
            parsed.loc[remaining] = pd.to_datetime(raw[remaining], errors="coerce")
    if tz_mask.any():
        aware = pd.to_datetime(raw[tz_mask], utc=True, errors="coerce")
        aware = aware.dt.tz_convert(local_tz).dt.tz_localize(None)
        parsed.loc[tz_mask] = aware

    return parsed


def load_folsom_data(irradiance_path, clear_sky_path, local_tz):
    irradiance = pd.read_csv(irradiance_path)
    clear_sky = pd.read_csv(clear_sky_path)

    irradiance["datetime"] = normalize_mixed_timezone(irradiance["datetime"], local_tz)
    irradiance = irradiance.dropna(subset=["datetime"])

    clear_sky["datetime"] = normalize_mixed_timezone(clear_sky["datetime"], local_tz)
    clear_sky = clear_sky.dropna(subset=["datetime"])

    irradiance = irradiance.rename(columns={"ghi": "ghi"})
    clear_sky = clear_sky.rename(columns={"ghi_clear": "ghi_clear"})

    merged = pd.merge(
        irradiance[["datetime", "ghi"]],
        clear_sky[["datetime", "ghi_clear"]],
        on="datetime",
        how="inner",
    )

    merged = merged.dropna(subset=["ghi", "ghi_clear"])
    merged = merged.sort_values("datetime")
    merged = merged.set_index("datetime")

    return merged


@click.command()
@click.option(
    "--irradiance-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to cleaned_irradiance.csv (needs datetime, ghi columns).",
)
@click.option(
    "--clear-sky-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to clear-sky CSV (needs datetime, ghi_clear columns).",
)
@click.option("--window-days", default=45, show_default=True, type=int)
@click.option("--ghi-col", default="ghi", show_default=True)
@click.option("--clear-ghi-col", default="ghi_clear", show_default=True)
@click.option("--local-tz", default="US/Pacific", show_default=True)
def main(irradiance_csv, clear_sky_csv, window_days, ghi_col, clear_ghi_col, local_tz):
    """Find optimal 45-day proxy window using irradiance and clear-sky CSVs."""
    df = load_folsom_data(irradiance_csv, clear_sky_csv, local_tz)

    if ghi_col not in df.columns or clear_ghi_col not in df.columns:
        raise click.UsageError(f"Missing required columns: {ghi_col} or {clear_ghi_col}.")

    start_date, end_date, _ = find_optimal_proxy_window(
        df,
        window_size=window_days,
        ghi_col=ghi_col,
        clear_ghi_col=clear_ghi_col,
    )

    proxy_dataset = df.loc[start_date : end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    print(f"Proxy dataset rows: {len(proxy_dataset)}")


if __name__ == "__main__":
    main()
