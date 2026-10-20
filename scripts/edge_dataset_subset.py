#!/usr/bin/env python3
"""List the files a device needs for scripts/edge_collaboration_benchmark.py, for ``rsync --files-from``.

The arm runner reads only the selected days (plus one warm-up day) of the prepared stream, so the device needs
the metadata files and those days' images, not the whole 60 GB directory.  Run on the host:

    python scripts/edge_dataset_subset.py --dataset data/folsom_prepared --days 7 > logs/edge_subset.txt
    rsync -a --files-from=logs/edge_subset.txt data/folsom_prepared/ pi5:continual-solar-irradiance/data/folsom_prepared/
"""

from __future__ import annotations

from pathlib import Path
import sys

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))


@click.command()
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--days", default=7, show_default=True)
def main(dataset: Path, days: int) -> None:
    from edge_collaboration_benchmark import select_days

    from cirr.data.loading import load_dataset

    selected = set(select_days(dataset, days))  # MMDD keys, same selection as the runner
    ds = load_dataset(dataset, split="test")
    names = set()
    for p in ds.periods:
        last = p["history"][-1]["image_name"]
        if last[4:8] in selected:
            names.update(h["image_name"] for h in p["history"])
    for meta in sorted(x.name for x in dataset.iterdir() if x.is_file()):
        print(meta)
    for n in sorted(names):
        print(f"images/{n}")
    print(f"# {len(selected)} days, {len(names)} images", file=sys.stderr)


if __name__ == "__main__":
    main()
