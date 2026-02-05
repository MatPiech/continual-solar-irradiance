from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import os
from pathlib import Path
import time

import click
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm


@click.command()
@click.option("--location", type=str, required=True, help="Location name (e.g., Alpnach)")
@click.option("--start", type=str, required=True, help="Start date (format: YYYY-MM-DD)")
@click.option("--end", type=str, required=True, help="End date (format: YYYY-MM-DD)")
@click.option(
    "--download-dir", type=click.Path(path_type=Path), default="skycam_data", help="Directory to save downloaded files"
)
@click.option("--timeout", type=int, default=60, help="Request timeout in seconds")
@click.option("--retries", type=int, default=2, help="Number of retries for failed requests")
def fetch_skycam(location: str, start: str, end: str, download_dir: Path, timeout: int, retries: int):
    """Download skycam files for a given period with 1 minute steps."""
    skycam_url = os.getenv("SKYCAM_URL")

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    auth = HTTPBasicAuth(os.getenv("USERNAME"), os.getenv("PASSWORD"))

    # Parse start and end dates
    try:
        start_time = datetime.strptime(start, "%Y-%m-%d")
        end_time = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1) - timedelta(minutes=1)
    except ValueError:
        print("Invalid date format. Use: YYYY-MM-DD")
        return

    if start_time > end_time:
        print("Start date must be before end date")
        return

    # Generate timestamps at 1-minute intervals
    print(f"Generating timestamps from {start} to {end}\n")
    timestamps = []
    current_time = start_time
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += timedelta(minutes=1)

    print(f"Generated {len(timestamps)} timestamps\n")

    def make_request(url, timeout, retries):
        """Make a request with retry logic."""
        for attempt in range(retries + 1):
            try:
                response = requests.get(url, auth=auth, timeout=timeout)
                return response
            except requests.exceptions.Timeout:
                if attempt < retries:
                    wait_time = 2**attempt  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    raise
            except requests.exceptions.RequestException:
                raise

    def download_file(file_info):
        """Download a single file and return result."""
        try:
            file_response = make_request(file_info["url"], timeout, retries)

            # Skip if file not found
            if file_response.status_code == 404:
                return (None, f"{file_info['name']} (not available)")

            if file_response.status_code == 200:
                # Create subdirectories as needed
                file_path = download_dir / file_info["relative_path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Write file with streaming for large files
                with file_path.open("wb") as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return (True, file_info["name"])
            else:
                return (None, f"{file_info['name']} (HTTP {file_response.status_code})")

        except requests.exceptions.Timeout:
            return (None, f"{file_info['name']} (timeout)")
        except requests.exceptions.RequestException as e:
            return (None, f"{file_info['name']} ({str(e)[:30]})")

    # Build file list for all timestamps
    print("Building file list...\n")
    all_files = []

    for ts in timestamps:
        year = ts.strftime("%Y")
        month = ts.strftime("%m")
        filename = f"{location}_{ts.strftime('%Y%m%d_%H-%M-%S')}.zip"

        file_url = f"{skycam_url}{location}/{year}/{month}/{filename}"
        relative_path = f"{location}/{year}/{month}/{filename}"

        all_files.append({"url": file_url, "name": filename, "relative_path": relative_path})

    if all_files:
        print(f"Prepared {len(all_files)} files to download\n")
        print("Downloading files...\n")

        downloaded_files = []
        skipped_files = []
        failed_files = []

        # Use threading for faster downloads
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(download_file, file_info) for file_info in all_files]

            for future in tqdm(as_completed(futures), total=len(all_files), desc="Downloading", unit="file"):
                status, msg = future.result()
                if status is True:
                    downloaded_files.append(msg)
                elif status is None:
                    skipped_files.append(msg)
                else:
                    failed_files.append(msg)

        print("\nDownload complete!")
        print(f"   Downloaded: {len(downloaded_files)} files")
        print(f"   Skipped (unavailable): {len(skipped_files)} files")
        print(f"   Failed: {len(failed_files)} files")
        print(f"\nFiles saved to {download_dir.absolute()}")

        if failed_files:
            print("\nFiles that failed to download:")
            for fname in failed_files[:10]:  # Show first 10
                print(f"   - {fname}")
            if len(failed_files) > 10:
                print(f"   ... and {len(failed_files) - 10} more")
    else:
        print("No files to download")


if __name__ == "__main__":
    load_dotenv()
    fetch_skycam()
