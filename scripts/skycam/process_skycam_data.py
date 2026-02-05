import csv
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import click
from tqdm import tqdm


@click.command()
@click.option(
    "--data-dir", type=click.Path(path_type=Path), default="skycam_data", help="Directory containing skycam zip files"
)
@click.option("--location", type=str, required=True, help="Location name (e.g., Alpnach)")
def process_skycam_data(data_dir: Path, location: str) -> None:
    """Process skycam zip files: extract images, rename them, and create irradiance CSV."""
    # Set default output directories
    output_dir = Path(location.lower()) / "images"

    csv_output = output_dir.parent / f"{location.lower()}_irradiance.csv"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all zip files
    zip_files = list(data_dir.rglob("*.zip"))

    if not zip_files:
        print(f"No zip files found in {data_dir}")
        return

    print(f"Found {len(zip_files)} zip files\n")
    print("Processing files...\n")

    # Data for CSV
    irradiance_data = []
    processed_count = 0
    error_count = 0

    # Process each zip file
    for zip_path in tqdm(zip_files, desc="Processing", unit="file"):
        try:
            # Extract timestamp from filename
            # Format: Location_YYYYMMDD_HH-MM-SS.zip
            filename = zip_path.stem
            timestamp_match = re.match(rf"{location}_(\d{{8}})_(\d{{2}})-(\d{{2}})-(\d{{2}})", filename)

            if not timestamp_match:
                continue

            date_part = timestamp_match.group(1)  # YYYYMMDD
            hour = timestamp_match.group(2)
            minute = timestamp_match.group(3)
            second = timestamp_match.group(4)

            # Format: YYYY_MM_DD__HH_MM_SS
            year = date_part[:4]
            month = date_part[4:6]
            day = date_part[6:8]
            formatted_name = f"{year}_{month}_{day}__{hour}_{minute}_{second}"

            # Open zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Look for the specific image file
                image_filename = f"{filename}_ExposureStack_Image_10_image.png"
                xml_filename = f"{filename}.xml"

                # Extract and rename image if it exists
                if image_filename in zip_ref.namelist():
                    # Extract to output directory with new name
                    image_data = zip_ref.read(image_filename)
                    output_image_path = output_dir / f"{formatted_name}.png"
                    with output_image_path.open("wb") as f:
                        f.write(image_data)

                # Extract irradiance from XML if it exists
                if xml_filename in zip_ref.namelist():
                    xml_data = zip_ref.read(xml_filename)

                    try:
                        # Parse XML (the files are UTF-8 despite claiming UTF-16 in declaration)
                        # Fix the XML declaration
                        xml_str = xml_data.decode("utf-8")
                        xml_str = xml_str.replace('encoding="utf-16"', 'encoding="utf-8"')
                        root = ET.fromstring(xml_str)

                        # Find Irradiance element
                        irradiance_elem = root.find(".//Pyranometer/Irradiance")

                        if irradiance_elem is not None:
                            irradiance = float(irradiance_elem.text)
                            irradiance_data.append({"datetime": formatted_name, "irradiance": irradiance})
                            processed_count += 1
                    except Exception:
                        error_count += 1
                        continue

        except Exception:
            error_count += 1
            continue

    # Write CSV file
    if irradiance_data:
        # Sort by datetime
        irradiance_data.sort(key=lambda x: x["datetime"])

        with csv_output.open("w", newline="") as csvfile:
            fieldnames = ["datetime", "irradiance"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in irradiance_data:
                writer.writerow(row)

        print("\nProcessing complete!")
        print(f"   Images saved to: {output_dir.absolute()}")
        print(f"   CSV saved to: {csv_output.absolute()}")
        print(f"   Total records: {len(irradiance_data)}")
        print(f"   Errors: {error_count}")
    else:
        print("\nNo data extracted")
        print(f"   Processed: {processed_count}")
        print(f"   Errors: {error_count}")


if __name__ == "__main__":
    process_skycam_data()
