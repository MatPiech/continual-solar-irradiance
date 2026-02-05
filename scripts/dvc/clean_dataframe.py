from pathlib import Path

import click
import pandas as pd
from PIL import Image
from tqdm import tqdm


@click.command()
@click.option("--images-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--irradiance-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-path", type=click.Path(path_type=Path), required=True)
@click.option("--timezone", type=str, default="US/Pacific")
@click.option("--date-col", type=str, default="timeStamp")
@click.option("--date-format", type=str, default="%Y-%m-%d %H:%M:%S")
@click.option("--image-format", type=str, default="jpg")
def clean_dataframe(
    images_path: Path,
    irradiance_path: Path,
    output_path: Path,
    timezone: str,
    date_col: str,
    date_format: str,
    image_format: str,
):
    df = pd.read_csv(irradiance_path)
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df = df.set_index(date_col)
    df.index.name = "datetime"
    df.index = df.index.tz_localize("UTC").tz_convert(timezone)
    existing_images = {
        pd.to_datetime(image_path.stem, format="%Y%m%d_%H%M%S").tz_localize(timezone, ambiguous="NaT")
        for image_path in images_path.rglob(f"*.{image_format}")
    }
    existing_images.discard(pd.NaT)  # Remove NaT values from ambiguous DST times
    print(f"Existing images: {len(existing_images)}")

    filtered_images = set()
    for image_name in tqdm(existing_images):
        image_path = Path(images_path, image_name.strftime("%Y%m%d_%H%M%S") + f".{image_format}")
        try:
            with Image.open(image_path) as img:
                img.verify()
        except (OSError, Exception) as e:
            print(f"Invalid image: {image_path} ({e})")
            continue

        filtered_images.add(image_name)

    print(f"Filtered images: {len(filtered_images)}")
    df = df.loc[df.index.isin(filtered_images)]
    df["image_name"] = df.index.strftime("%Y%m%d_%H%M%S") + f".{image_format}"
    print(f"Data samples: {len(df)}")
    df.to_csv(output_path)


if __name__ == "__main__":
    clean_dataframe()
