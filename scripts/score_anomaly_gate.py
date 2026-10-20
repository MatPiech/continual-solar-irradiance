"""Score a stream with an already-fitted admissibility gate.

Uses the forest and threshold written by ``scripts/fit_anomaly_gate.py`` unchanged,
so a site other than the one it was fitted on is scored against the source
distribution -- which is what makes the flag rate a measure of deployment shift.
With ``--corruption-schedule`` the stream carries synthetic faults first, and the
scores are the ones the gate would see on the corrupted frames.

    python scripts/score_anomaly_gate.py --gate results/ch4/gate/folsom \\
        --stream data/karpicko_prepared --out results/ch4/gate/karpicko
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from cirr.evaluation.admissibility import CorruptedStream, anomaly_score, embed, load_gate, stamp
from cirr.evaluation.campaign import SOURCE_CHECKPOINT, load_campaign_days
from cirr.evaluation.runner import load_eval_model
from cirr.utils.provenance import git_commit, set_determinism


@click.command()
@click.option("--gate", "gate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--stream", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--corruption-schedule", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=str(SOURCE_CHECKPOINT), show_default=True)
@click.option("--workers", type=click.IntRange(min=0), default=8, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True)
def main(
    gate_dir: Path, stream: Path, corruption_schedule: Path | None, checkpoint: Path, workers: int, device: str, out_dir: Path
) -> None:
    set_determinism(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    forest, meta = load_gate(gate_dir)
    model = load_eval_model(Path(checkpoint), device)

    data = load_campaign_days(stream, min_ghi_clear=0.001)
    dataset = data.dataset
    if corruption_schedule is not None:
        dataset = CorruptedStream(dataset, pd.read_csv(corruption_schedule, dtype={"timestamp": str}))
    idx = [i for d in data.days for i in data.day_map[d]]
    scores = anomaly_score(forest, embed(model, dataset, idx, device, workers))

    frame = pd.DataFrame({"timestamp": [stamp(data.dataset.periods[i]) for i in idx], "score": scores})
    frame["flagged"] = frame.score > meta["threshold"]
    frame.to_csv(out_dir / "scores.csv", index=False)
    frame["month"] = frame.timestamp.str[4:6]
    summary = {
        "gate": str(gate_dir),
        "threshold": meta["threshold"],
        "percentile": meta["percentile"],
        "stream": str(stream),
        "corruption_schedule": str(corruption_schedule) if corruption_schedule else None,
        "n_stream": len(frame),
        "stream_flag_rate_pct": float(100 * frame.flagged.mean()),
        "stream_flag_rate_by_month_pct": (100 * frame.groupby("month").flagged.mean()).round(2).to_dict(),
        "git_commit": git_commit(),
    }
    # the policy's --anomaly-gate reads the threshold from here, so a scored stream is a usable gate
    (out_dir / "gate.json").write_text(json.dumps({**meta, **summary}, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_stream", "stream_flag_rate_pct")}, indent=2))


if __name__ == "__main__":
    main()
