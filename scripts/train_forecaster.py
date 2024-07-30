import os
from pathlib import Path

import click
import dvc.api
import lightning.pytorch as pl
import torch
from dotenv import find_dotenv, load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

from continual_solar_irradiance.datamodules.forecasting import ForecastingDataModule
from continual_solar_irradiance.models.forecaster import Forecaster
from continual_solar_irradiance.utils import utils

log = utils.get_logger(__name__)


@click.command()
@click.option('--data-root', type=click.Path(exists=True, path_type=Path), required=True)
def train_forecaster(data_root: Path):
    load_dotenv(find_dotenv('.env'))

    cfg = OmegaConf.create(dvc.api.params_show()['train_forecaster'])

    pl.seed_everything(seed=cfg.seed)

    datamodule = ForecastingDataModule(
        root_data_path=data_root,
        periods_path=data_root / 'periods.pickle',
        augment=cfg.datamodule.augment,
        image_size=cfg.datamodule.image_size,
        image_mean=cfg.datamodule.image_mean,
        image_std=cfg.datamodule.image_std,
        batch_size=cfg.datamodule.batch_size,
        workers=cfg.datamodule.workers,
        seed=cfg.seed,
    )

    model = Forecaster(
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        input_channels=cfg.model.channels,
        loss_function=cfg.model.loss_function,
        lr=cfg.model.lr,
        lr_patience=cfg.model.lr_patience,
    )

    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.model_checkpoint)
    model_summary_callback = ModelSummary(max_depth=1)
    early_stopping_callback = EarlyStopping(**cfg.callbacks.early_stopping)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [
        checkpoint_callback,
        model_summary_callback,
        early_stopping_callback,
    ]

    if not cfg.debug:
        logger = NeptuneLogger(
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            project=os.environ['NEPTUNE_PROJECT'],
            log_model_checkpoints=True,
        )
        callbacks.append(lr_monitor)

        logger.log_hyperparams(cfg.datamodule)
    else:
        logger = None

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices=cfg.trainer.devices if not None else -1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
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
            log.info('Compiling model')
            model = torch.compile(model)

        log.info('Starting training process')
        trainer.fit(model, datamodule)

        log.info('Starting testing process for the best checkpoint')
        trainer.test(model, datamodule, ckpt_path='best')
        log.info(f'Best model checkpoint: {trainer.checkpoint_callback.best_model_path}')
    else:
        assert cfg.restore_from_ckpt is not None
        log.info(f'Starting testing process for {cfg.restore_from_ckpt} checkpoint')
        trainer.test(model, datamodule, ckpt_path=cfg.restore_from_ckpt)


if __name__ == '__main__':
    train_forecaster()
