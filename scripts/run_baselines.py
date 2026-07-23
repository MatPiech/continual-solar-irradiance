"""Orchestrate baseline comparisons."""

from __future__ import annotations

from pathlib import Path
import subprocess

import click

BASELINES = {
    "static": ["python", "scripts/evaluate.py"],
    "ungated_ogd": ["python", "scripts/evaluate_paper_lagged_ogd.py"],
    "gated": ["python", "scripts/evaluate_gated.py"],
    "fedavg": ["python", "scripts/fl_benchmark.py", "--algorithm", "fedavg"],
    "gated_fl": ["python", "scripts/fl_benchmark.py", "--algorithm", "gated"],
}


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--dataset-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--baseline",
    type=click.Choice(list(BASELINES.keys())),
    multiple=True,
    default=["static", "ungated_ogd", "gated"],
    show_default=True,
)
@click.option("--dry-run", is_flag=True, default=False)
def main(checkpoint: Path, dataset_path: Path, baseline: tuple[str, ...], dry_run: bool) -> None:
    """Run selected baselines sequentially."""
    for name in baseline:
        cmd = BASELINES[name] + [
            "--checkpoint",
            str(checkpoint),
            "--dataset-path",
            str(dataset_path),
        ]
        if name in ("fedavg", "gated_fl"):
            parent = dataset_path.parent
            cmd = BASELINES[name] + [
                "--checkpoint",
                str(checkpoint),
                "--dataset",
                str(parent / "folsom_prepared"),
                "--dataset",
                str(parent / "karpicko_prepared"),
                "--dataset",
                str(dataset_path),
            ]
        print(f"\n>>> Running baseline: {name}\n    {' '.join(cmd)}")
        if dry_run:
            continue
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
