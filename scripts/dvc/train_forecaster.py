from pathlib import Path

import click
import dvc.api
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
import torch
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

from cirr.data.forecasting import ForecastingDataModule
from cirr.data.transforms import input_channels, normalize_optical_flow
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_image_encoder_checkpoint
from cirr.utils import utils

log = utils.get_logger(__name__)


@click.command()
@click.option("--data-root", type=click.Path(exists=True, path_type=Path), required=True)
def train_forecaster(data_root: Path):
    cfg = OmegaConf.create(dvc.api.params_show()["train_forecaster"])

    pl.seed_everything(seed=cfg.seed)

    optical_flow = normalize_optical_flow(cfg.datamodule.optical_flow)

    datamodule = ForecastingDataModule(
        root_data_path=data_root,
        periods_path=data_root / "train_periods.pickle",
        augment=cfg.datamodule.augment,
        image_size=cfg.datamodule.image_size,
        image_mean=cfg.datamodule.image_mean,
        image_std=cfg.datamodule.image_std,
        batch_size=cfg.datamodule.batch_size,
        workers=cfg.datamodule.workers,
        seed=cfg.seed,
        optical_flow=optical_flow,
    )

    encoder_ckpt = cfg.model.get("encoder_checkpoint")
    use_pretrained = cfg.model.pretrained and not encoder_ckpt

    model = Forecaster(
        model_name=cfg.model.model_name,
        pretrained=use_pretrained,
        input_channels=input_channels(optical_flow=optical_flow),
        loss_function=cfg.model.loss_function,
        lr=cfg.model.lr,
        lr_patience=cfg.model.lr_patience,
    )

    if encoder_ckpt:
        load_image_encoder_checkpoint(model, encoder_ckpt)
        log.info(f"Loaded image encoder weights from {encoder_ckpt}")

    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.model_checkpoint)
    model_summary_callback = ModelSummary(max_depth=1)
    early_stopping_callback = EarlyStopping(**cfg.callbacks.early_stopping)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [
        checkpoint_callback,
        model_summary_callback,
        early_stopping_callback,
    ]

    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("enabled", False):
        logger = MLFlowLogger(
            experiment_name=mlflow_cfg.get("experiment_name", "forecaster"),
            tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"),
            run_name=mlflow_cfg.get("run_name") or None,
        )
        callbacks.append(lr_monitor)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    else:
        logger = None

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices=cfg.trainer.devices if not None else -1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=DDPStrategy(
            ddp_comm_wrapper=default.fp16_compress_wrapper,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            static_graph=True,
        ),
        precision=cfg.trainer.precision,
        min_epochs=cfg.trainer.min_epochs,
        max_epochs=cfg.trainer.max_epochs,
        benchmark=True,
        sync_batchnorm=cfg.trainer.devices > 0,
        check_val_every_n_epoch=1,
    )

    if not cfg.test_only:
        if cfg.model.compile:
            log.info("Compiling model")
            model = torch.compile(model)

        log.info("Starting training process")
        trainer.fit(model, datamodule)

        log.info("Starting testing process for the best checkpoint")
        trainer.test(model, datamodule, ckpt_path="best")
        log.info(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        assert cfg.restore_from_ckpt is not None
        log.info(f"Starting testing process for {cfg.restore_from_ckpt} checkpoint")
        trainer.test(model, datamodule, ckpt_path=cfg.restore_from_ckpt)


if __name__ == "__main__":
    train_forecaster()
