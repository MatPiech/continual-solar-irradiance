from datetime import date, timedelta
from pathlib import Path

from astral import LocationInfo
from astral.sun import dawn, dusk, sunrise, sunset
import click
import pandas as pd
import pytz
from timezonefinder import TimezoneFinder

# Depression angles for different twilight types
ASTRO_DEPRESSION = 18.0
NAUTICAL_DEPRESSION = 12.0
CIVIL_DEPRESSION = 6.0


def calculate_all_twilight(
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Calculates all types of twilight, sunrise, and sunset for a specific location over a date range.

    This function iterates through the provided date range and uses the
    astral library to calculate:
    - Astronomical twilight (18 degrees below horizon)
    - Nautical twilight (12 degrees below horizon)
    - Civil twilight (6 degrees below horizon)
    - Sunrise and sunset (0 degrees)

    Parameters
    ----------
    latitude : float
        The latitude of the location in degrees (positive for North, negative for South).
    longitude : float
        The longitude of the location in degrees (positive for East, negative for West).
    timezone : str
        The timezone of the location (e.g., 'Europe/Warsaw').
    start_date : datetime.date
        The start date of the period (inclusive).
    end_date : datetime.date
        The end date of the period (inclusive).

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the following columns:
        - date: The date of the event.
        - astronomical_dawn: The UTC datetime of astronomical dawn.
        - nautical_dawn: The UTC datetime of nautical dawn.
        - civil_dawn: The UTC datetime of civil dawn.
        - sunrise: The UTC datetime of sunrise.
        - sunset: The UTC datetime of sunset.
        - civil_dusk: The UTC datetime of civil dusk.
        - nautical_dusk: The UTC datetime of nautical dusk.
        - astronomical_dusk: The UTC datetime of astronomical dusk.

        Note: If twilight does not occur (e.g., during summer
        nights at high latitudes), the value will be NaT (Not a Time).
    """

    # Create a LocationInfo object
    location = LocationInfo(name="Target", region="Region", timezone=timezone, latitude=latitude, longitude=longitude)

    data = []
    current_date = start_date

    while current_date <= end_date:
        day_record = {"date": current_date}

        # Calculate dawn events
        for event_name, depression in [
            ("astronomical_dawn", ASTRO_DEPRESSION),
            ("nautical_dawn", NAUTICAL_DEPRESSION),
            ("civil_dawn", CIVIL_DEPRESSION),
        ]:
            try:
                day_record[event_name] = dawn(location.observer, current_date, depression)
            except ValueError:
                day_record[event_name] = pd.NaT

        # Calculate sunrise
        try:
            day_record["sunrise"] = sunrise(location.observer, current_date)
        except ValueError:
            day_record["sunrise"] = pd.NaT

        # Calculate sunset
        try:
            day_record["sunset"] = sunset(location.observer, current_date)
        except ValueError:
            day_record["sunset"] = pd.NaT

        # Calculate dusk events
        for event_name, depression in [
            ("civil_dusk", CIVIL_DEPRESSION),
            ("nautical_dusk", NAUTICAL_DEPRESSION),
            ("astronomical_dusk", ASTRO_DEPRESSION),
        ]:
            try:
                day_record[event_name] = dusk(location.observer, current_date, depression)
            except ValueError:
                day_record[event_name] = pd.NaT

        data.append(day_record)
        current_date += timedelta(days=1)

    return pd.DataFrame(data)


def get_timezone_from_coordinates(latitude: float, longitude: float) -> str:
    """
    Automatically determine timezone from latitude and longitude coordinates.

    Parameters
    ----------
    latitude : float
        The latitude of the location.
    longitude : float
        The longitude of the location.

    Returns
    -------
    str
        The timezone name (e.g., 'Europe/Warsaw').
    """
    tf = TimezoneFinder()
    timezone = tf.timezone_at(lat=latitude, lng=longitude)

    if timezone is None:
        raise ValueError(f"Could not determine timezone for coordinates ({latitude}, {longitude})")

    return timezone


def generate_csv(
    output_path: Path,
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: date,
    end_date: date,
) -> None:
    """
    Generates a CSV file containing all twilight types, sunrise, and sunset times.

    Parameters
    ----------
    output_path : Path
        The path and name of the output CSV file (e.g., 'output.csv').
    latitude : float
        The latitude of the location.
    longitude : float
        The longitude of the location.
    timezone : str
        The timezone of the location.
    start_date : datetime.date
        The start date for the data generation.
    end_date : datetime.date
        The end date for the data generation.

    Returns
    -------
    None
        The function saves a file to disk and does not return a value.
    """
    print(f"Calculating astronomical data for location ({latitude}, {longitude})...")
    print(f"Using timezone: {timezone}")

    df = calculate_all_twilight(latitude, longitude, timezone, start_date, end_date)

    # Convert from UTC to local timezone and format as HH:MM:SS
    local_tz = pytz.timezone(timezone)
    time_columns = [
        "astronomical_dawn",
        "nautical_dawn",
        "civil_dawn",
        "sunrise",
        "sunset",
        "civil_dusk",
        "nautical_dusk",
        "astronomical_dusk",
    ]

    for col in time_columns:
        df[col] = df[col].dt.tz_convert(local_tz).dt.strftime("%H:%M:%S")

    df.to_csv(output_path, index=False)

    print(f"Success! Data saved to '{output_path!s}'.")
    print(f"Rows generated: {len(df)}")


@click.command()
@click.option(
    "--latitude", "-lat", type=float, required=True, help="Latitude of the location (positive for North, negative for South)"
)
@click.option(
    "--longitude", "-lon", type=float, required=True, help="Longitude of the location (positive for East, negative for West)"
)
@click.option(
    "--start-date", "-s", type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date in YYYY-MM-DD format"
)
@click.option(
    "--end-date", "-e", type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date in YYYY-MM-DD format"
)
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output CSV file path")
@click.option(
    "--timezone",
    "-tz",
    type=str,
    default=None,
    help="Timezone (optional, will be auto-detected from coordinates if not provided)",
)
def main(latitude, longitude, start_date, end_date, output, timezone):
    """Calculate astronomical twilight, sunrise, and sunset times for a location."""
    # Auto-detect timezone if not provided
    if timezone is None:
        timezone = get_timezone_from_coordinates(latitude, longitude)

    # Convert datetime to date
    start = start_date.date()
    end = end_date.date()

    generate_csv(output, latitude, longitude, timezone, start, end)


if __name__ == "__main__":
    main()
