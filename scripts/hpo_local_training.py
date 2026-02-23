from pathlib import Path
import sys
from typing import Any

import click
import optuna
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from local_training import run_local_training  # noqa: E402


def _build_pruner(name: str, max_resource: int) -> optuna.pruners.BasePruner:
    min_resource = min(5, max_resource)
    if name == "asha":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=3,
        )
    raise click.BadParameter("pruner must be 'asha' or 'hyperband'")


def _build_sampler() -> optuna.samplers.BaseSampler:
    return optuna.samplers.TPESampler(multivariate=True, seed=42)


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint file",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to prepared dataset directory",
)
@click.option("--num-days", type=int, default=45, show_default=True)
@click.option("--start-day", type=str, default="0101", show_default=True)
@click.option("--adapt-batch-size", type=int, default=32, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--freeze-image-encoder", is_flag=True, default=False)
@click.option(
    "--trials",
    type=int,
    default=50,
    show_default=True,
    help="Number of HPO trials to run",
)
@click.option(
    "--pruner",
    type=click.Choice(["asha", "hyperband"], case_sensitive=False),
    default="hyperband",
    show_default=True,
)
@click.option(
    "--output-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("./hpo_results.csv"),
    show_default=True,
)
def main(
    checkpoint: Path,
    dataset_path: Path,
    num_days: int,
    start_day: str,
    adapt_batch_size: int,
    seed: int,
    freeze_image_encoder: bool,
    trials: int,
    pruner: str,
    output_csv: Path,
) -> None:
    """Run Optuna HPO with ASHA/Hyperband pruning for local training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pruner_impl = _build_pruner(pruner.lower(), num_days)
    sampler = _build_sampler()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner_impl)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "adapt_lr": trial.suggest_float("adapt_lr", 1e-5, 1e-4, log=True),
            "adapt_weight_decay": trial.suggest_float("adapt_weight_decay", 1e-5, 0.3, log=True),
            "adapt_epochs": trial.suggest_int("adapt_epochs", 1, 3),
            "adapt_replay_buffer_size": trial.suggest_int("adapt_replay_buffer_size", 10000, 50000),
            "adapt_replay_ratio": trial.suggest_float("adapt_replay_ratio", 0.0, 2.0),
            "adapt_l2sp_lambda": trial.suggest_float("adapt_l2sp_lambda", 1e-6, 1e-3, log=True),
        }

        def progress_callback(day_idx: int, metrics: dict[str, float]) -> None:
            trial.report(metrics["rmse"], step=day_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, _, overall_metrics, _ = run_local_training(
            checkpoint_path=checkpoint,
            dataset_path=dataset_path,
            num_days=num_days,
            start_day=start_day,
            adapt_lr=params["adapt_lr"],
            adapt_weight_decay=params["adapt_weight_decay"],
            adapt_max_grad_norm=1.0,
            adapt_l2sp_lambda=params["adapt_l2sp_lambda"],
            adapt_batch_size=adapt_batch_size,
            adapt_epochs=params["adapt_epochs"],
            adapt_replay_buffer_size=params["adapt_replay_buffer_size"],
            adapt_replay_ratio=params["adapt_replay_ratio"],
            device=device,
            seed=seed,
            freeze_image_encoder=freeze_image_encoder,
            progress_callback=progress_callback,
        )

        return overall_metrics["rmse"]

    study.optimize(objective, n_trials=trials)

    results_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    best = study.best_trial
    click.echo(f"Best RMSE: {best.value:.6f}")
    click.echo(f"Best params: {best.params}")
    click.echo(f"Saved results to {output_csv}")


if __name__ == "__main__":
    main()
