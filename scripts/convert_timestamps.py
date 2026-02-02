from pathlib import Path

import click
import pandas as pd
import pytz
from tqdm import tqdm


@click.command()
@click.option("--data-raw", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-path", type=click.Path(path_type=Path), required=True)
@click.option("--timezone", type=str, default="US/Pacific")
def convert_timestamps(data_raw: Path, output_path: Path, timezone: str) -> None:
    output_path.mkdir(exist_ok=True, parents=True)

    local_timezone = pytz.timezone(timezone)
    utc = pytz.utc

    for image_path in tqdm(sorted(data_raw.rglob("*.jpg"))):
        filename = image_path.stem.replace("_", "")[:14]  # YYYYMMDDHHMMSS
        date = pd.to_datetime(filename, format="%Y%m%d%H%M%S").round("1min")
        utc_date = utc.localize(date)
        local_timestamp = utc_date.astimezone(local_timezone)
        filename = local_timestamp.strftime("%Y%m%d_%H%M%S") + ".jpg"

        image_path.rename(output_path / filename)


if __name__ == "__main__":
    convert_timestamps()
