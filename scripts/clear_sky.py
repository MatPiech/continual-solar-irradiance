from pathlib import Path

import click
import pandas as pd
import pvlib


@click.command()
@click.option("--latitude", type=float, help="Latitude of the location.")
@click.option("--longitude", type=float, help="Longitude of the location.")
@click.option("--timezone", type=str, help="Timezone of the location.")
@click.option("--start", type=str, help="Start time (local). Format `YYYY-MM-DD HH:MM:SS`")
@click.option("--end", type=str, help="End time (local). Format `YYYY-MM-DD HH:MM:SS`")
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default="data/prepared/clear_sky.csv",
    help="Output CSV file path for clear-sky data.",
)
def generate_clear_sky_reference(
    latitude: float, longitude: float, timezone: str, start: str, end: str, output_path: str
) -> None:
    print(f"Generating clear-sky data for ({latitude}, {longitude}) using Ineichen-Perez model.")
    print(f"Time range (local): {start} to {end}")

    # Create the 1-minute time index spanning the specified time range
    start = pd.Timestamp(start, tz=timezone)
    end = pd.Timestamp(end, tz=timezone)
    times = pd.date_range(start, end, freq="1min", tz=timezone)

    # Create a pvlib.Location object
    location = pvlib.location.Location(latitude, longitude, tz=timezone)

    # Get solar position (zenith, azimuth, etc.) for the times
    solar_pos = location.get_solarposition(times)

    # Get atmospheric data (Linke Turbidity) from pvlib's built-in dataset
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)

    # Calculate clear-sky GHI using the Ineichen-Perez model
    clearsky = location.get_clearsky(times, model="ineichen", solar_position=solar_pos, linke_turbidity=linke_turbidity)

    # Rename 'ghi' to 'ghi_clear' for clarity.
    clearsky_data = clearsky.rename(columns={"ghi": "ghi_clear"})
    clearsky_data.drop(columns=["dni", "dhi"], inplace=True)

    # Ensure nighttime GHI is 0 and not NaN
    clearsky_data["ghi_clear"] = clearsky_data["ghi_clear"].fillna(0).clip(lower=0)

    # Remove timezone info for saving
    clearsky_data.index = clearsky_data.index.tz_localize(None)

    print("\nSuccessfully generated local clear-sky data using Ineichen model.")
    print("\n--- Data Head ---")
    print(clearsky_data.head(3))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clearsky_data.to_csv(output_path, index_label="datetime")
    print(f"\nClear-sky data saved to '{output_path}'")


if __name__ == "__main__":
    generate_clear_sky_reference()
