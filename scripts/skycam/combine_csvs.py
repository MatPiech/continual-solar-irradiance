"""Script to combine multiple CSV files with unique datetimes."""

from pathlib import Path

import click
import pandas as pd


@click.command()
@click.argument("csv_files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path, writable=True),
    default="combined.csv",
    help="Output CSV file path (default: combined.csv)",
)
@click.option("--datetime-col", default="datetime", help="Name of datetime column (default: datetime)")
@click.option(
    "--keep",
    type=click.Choice(["first", "last"]),
    default="first",
    help="Which duplicate datetime to keep - first or last (default: first)",
)
def combine_csvs(csv_files: tuple[Path, ...], output: Path, datetime_col: str, keep: str):
    """
    Combine multiple CSV files into a single file with unique datetimes.

    CSV_FILES: One or more CSV file paths to combine
    """
    if not csv_files:
        print("Error: Please provide at least one CSV file.")
        return

    try:
        # Read all CSV files
        dfs = []
        for file_path in csv_files:
            print(f"Reading {file_path}...")
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"  Loaded {len(df)} rows")

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal rows before deduplication: {len(combined_df)}")

        # Remove duplicate datetimes
        if datetime_col not in combined_df.columns:
            print(f"Error: Column '{datetime_col}' not found in CSV files.")
            print(f"Available columns: {', '.join(combined_df.columns)}")
            return

        combined_df = combined_df.drop_duplicates(subset=[datetime_col], keep=keep)
        print(f"Total rows after deduplication: {len(combined_df)}")

        # Sort by datetime
        combined_df = combined_df.sort_values(datetime_col).reset_index(drop=True)

        # Save to output file
        combined_df.to_csv(output, index=False)
        print(f"\nSuccessfully saved combined CSV to: {output}")

    except Exception as e:
        print(f"Error: {e!s}")
        raise


if __name__ == "__main__":
    combine_csvs()
