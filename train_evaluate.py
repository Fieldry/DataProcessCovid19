import os
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pandas as pd

from configs.config import ml_example_params as params
from datasets.loader import EhrDataModule, get_los_info
from pipelines import DlPipeline, MlPipeline


def run_ml_experiment(config):
    los_config = get_los_info(
        f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(
        f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # logger
    checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    logger = CSVLogger(
        save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)
    L.seed_everything(config["seed"])  # seed for reproducibility

    # train/val/test
    pipeline = MlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1,
                        logger=logger, num_sanity_val_steps=0)
    trainer.fit(pipeline, dm)

    trainer.test(pipeline, dm)
    perf = pipeline.test_performance
    outputs = pipeline.test_outputs
    return perf, outputs


def run_dl_experiment(config):
    los_config = get_los_info(
        f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(
        f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # logger
    checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    logger = CSVLogger(
        save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)

    # EarlyStop and checkpoint callback
    dirpath = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints'
    if config["task"] in ["outcome", "multitask"]:
        early_stopping_callback = EarlyStopping(
            monitor=config["main_metric"], patience=config["patience"], mode="max")
        checkpoint_callback = ModelCheckpoint(
            filename="best", monitor=config["main_metric"], mode="max", dirpath=dirpath, enable_version_counter=False)
    elif config["task"] == "los":
        early_stopping_callback = EarlyStopping(
            monitor=config["main_metric"], patience=config["patience"], mode="min")
        checkpoint_callback = ModelCheckpoint(
            filename="best", monitor=config["main_metric"], mode="min", dirpath=dirpath, enable_version_counter=False)

    L.seed_everything(config["seed"])  # seed for reproducibility

    # train/val/test
    pipeline = DlPipeline(config, dm)
    if config["accelerator"] == "cpu":
        devices = "auto"
    else:
        devices = config["devices"]
    trainer = L.Trainer(accelerator=config["accelerator"], devices=devices, max_epochs=config["epochs"], logger=logger, callbacks=[
                        early_stopping_callback, checkpoint_callback], num_sanity_val_steps=1)
    trainer.fit(pipeline, dm)

    checkpoint_path = checkpoint_callback.best_model_path
    trainer.test(pipeline, dm, ckpt_path=checkpoint_path)
    perf = pipeline.test_performance
    outputs = pipeline.test_outputs

    return perf, outputs


if __name__ == "__main__":
    performance_table = {'dataset': [], 'task': [], 'model': [], 'fold': [], 'auprc': [], 'auroc': [
    ], 'es': [], 'accuracy': [], 'f1': [], 'mae': [], 'mse': [], 'osmae': [], 'rmse': [], 'r2': []}
    for config in params:
        run_func = run_ml_experiment if config["model"] in [
            "RF", "DT", "GBDT", "CatBoost", "XGBoost"] else run_dl_experiment
        for fold in range(1, 10):
            config['fold'] = fold
            print(config)
            perf, outputs = run_func(config)

            performance_table['dataset'].append(config['dataset'])
            performance_table['task'].append(config['task'])
            performance_table['model'].append(config['model'])
            performance_table['fold'].append(fold)
            if config['task'] == 'outcome':
                performance_table['accuracy'].append(perf['accuracy'])
                performance_table['auroc'].append(perf['auroc'])
                performance_table['auprc'].append(perf['auprc'])
                performance_table['f1'].append(perf['f1'])
                performance_table['es'].append(perf['es'])
                performance_table['mae'].append(None)
                performance_table['mse'].append(None)
                performance_table['rmse'].append(None)
                performance_table['r2'].append(None)
                performance_table['osmae'].append(None)
            elif config['task'] == 'los':
                performance_table['accuracy'].append(None)
                performance_table['auroc'].append(None)
                performance_table['auprc'].append(None)
                performance_table['f1'].append(None)
                performance_table['es'].append(None)
                performance_table['mae'].append(perf['mae'])
                performance_table['mse'].append(perf['mse'])
                performance_table['rmse'].append(perf['rmse'])
                performance_table['r2'].append(perf['r2'])
                performance_table['osmae'].append(None)
            else:
                performance_table['accuracy'].append(perf['accuracy'])
                performance_table['auroc'].append(perf['auroc'])
                performance_table['auprc'].append(perf['auprc'])
                performance_table['f1'].append(perf['f1'])
                performance_table['es'].append(perf['es'])
                performance_table['mae'].append(perf['mae'])
                performance_table['mse'].append(perf['mse'])
                performance_table['rmse'].append(perf['rmse'])
                performance_table['r2'].append(perf['r2'])
                performance_table['osmae'].append(perf['osmae'])

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(performance_table).to_csv(
        f'results/performance.csv', index=False)
