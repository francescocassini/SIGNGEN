import json
import os
import time
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pathlib import Path
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from mGPT.callback import build_callbacks
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


class InferenceProgressLogger(pl.Callback):
    """Text progress logs for test/inference, useful when TTY progress bars are not visible."""

    def __init__(self, logger, log_every_n_batches=50):
        super().__init__()
        self.logger = logger
        self.log_every_n_batches = max(1, int(log_every_n_batches))
        self._start_time = None
        self._last_log_time = None
        self._seen_batches = 0
        self._total_batches = None

    @staticmethod
    def _format_time(seconds):
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def on_test_start(self, trainer, pl_module):
        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._seen_batches = 0

        raw_total = getattr(trainer, "num_test_batches", None)
        if isinstance(raw_total, (list, tuple)):
            self._total_batches = int(sum(raw_total))
        elif raw_total is None:
            self._total_batches = None
        else:
            self._total_batches = int(raw_total)

        dataset_size = "unknown"
        sample_cap = None
        try:
            dataset_size = len(trainer.datamodule.test_dataset)
            sample_cap = getattr(trainer.datamodule.cfg.TEST, "MAX_SAMPLES", None)
        except Exception:
            pass
        self.logger.info(
            "Inference started | test_samples=%s | sample_cap=%s | test_batches=%s | batch_size=%s",
            dataset_size,
            sample_cap if sample_cap is not None else "none",
            self._total_batches if self._total_batches is not None else "unknown",
            trainer.datamodule.cfg.TEST.BATCH_SIZE if hasattr(trainer.datamodule, "cfg") else "unknown",
        )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._seen_batches += 1
        now = time.time()

        should_log = (
            self._seen_batches == 1
            or self._seen_batches % self.log_every_n_batches == 0
            or (self._total_batches is not None and self._seen_batches >= self._total_batches)
        )
        if not should_log:
            return

        elapsed = max(now - self._start_time, 1e-6)
        bps = self._seen_batches / elapsed
        if self._total_batches is not None and self._total_batches > 0:
            pct = 100.0 * self._seen_batches / self._total_batches
            remaining_batches = max(self._total_batches - self._seen_batches, 0)
            eta = remaining_batches / max(bps, 1e-9)
            self.logger.info(
                "Inference progress | %d/%d batches (%.2f%%) | %.2f batch/s | elapsed=%s | ETA=%s",
                self._seen_batches,
                self._total_batches,
                pct,
                bps,
                self._format_time(elapsed),
                self._format_time(eta),
            )
        else:
            self.logger.info(
                "Inference progress | batches_done=%d | %.2f batch/s | elapsed=%s",
                self._seen_batches,
                bps,
                self._format_time(elapsed),
            )
        self._last_log_time = now

    def on_test_end(self, trainer, pl_module):
        elapsed = 0.0 if self._start_time is None else (time.time() - self._start_time)
        self.logger.info(
            "Inference finished | total_batches=%d | total_time=%s",
            self._seen_batches,
            self._format_time(elapsed),
        )


def print_table(title, metrics, logger=None):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")

    logger.info(metrics) if logger else None


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.USE_GPUS
    cfg.FOLDER = cfg.TEST.FOLDER

    # for data_cfg in [h2s_cfg, csl_cfg]:
        # cfg['DATASET']['H2S'] = data_cfg
    # Logger
    logger = create_logger(cfg, phase="test")
    # logger.info(OmegaConf.to_yaml(cfg))
    # logger.info(data_cfg['DATASET_NAME'])

    if cfg.ACCELERATOR == "gpu" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        cfg.ACCELERATOR = "cpu"
        cfg.DEVICE = 1
        cfg.NUM_NODES = 1
        cfg.TRAIN.NUM_WORKERS = 0
        cfg.EVAL.NUM_WORKERS = 0
        cfg.TEST.NUM_WORKERS = 0

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase="test")
    callbacks.append(InferenceProgressLogger(logger=logger, log_every_n_batches=50))
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))
    if len(datamodule.test_dataset) == 0:
        h2s_root = os.environ.get("SOKE_H2S_ROOT", "../data/How2Sign")
        csl_root = os.environ.get("SOKE_CSL_ROOT", "../data/CSL-Daily")
        pho_root = os.environ.get("SOKE_PHOENIX_ROOT", "../data/Phoenix_2014T")
        raise RuntimeError(
            "No test samples found. Populate pose directories under "
            f"{h2s_root}, {csl_root}/poses, {pho_root}."
        )

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    if isinstance(cfg.DEVICE, int):
        num_devices = cfg.DEVICE
    else:
        try:
            num_devices = len(cfg.DEVICE)
        except TypeError:
            num_devices = int(cfg.DEVICE)

    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true" if num_devices > 1 else 'auto',
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
    )

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # loading state dict
    if not cfg.TEST.CHECKPOINTS:
        ckpt_folder = os.path.join(cfg.FOLDER_EXP.replace('results', 'experiments'), 'checkpoints')
        cfg.TEST.CHECKPOINTS = os.path.join(ckpt_folder, 'last.ckpt')
    load_pretrained(cfg, model, logger, phase="test")
    cfg.TIME = cfg.TEST.CHECKPOINTS.split('/')[-1]

    # Calculate metrics
    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES

    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        logger.info(f"Evaluating {metrics_type} - Replication {i}")
        metrics = trainer.test(model, datamodule=datamodule)[0]
        # if "TM2TMetrics" in metrics_type and cfg.model.params.task == "t2m" and cfg.model.params.stage != 'vae':
        #     # mm meteics
        #     logger.info(f"Evaluating MultiModality - Replication {i}")
        #     datamodule.mm_mode(True)
        #     mm_metrics = trainer.test(model, datamodule=datamodule)[0]
        #     # metrics.update(mm_metrics)
        #     metrics.update(mm_metrics)
        #     datamodule.mm_mode(False)
        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    all_metrics_new = {}

    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item),
                                                    replication_times)
        all_metrics_new[key + "/mean"] = f"{mean:.3f}"
        # all_metrics_new[key + "/conf_interval"] = conf_interval

    print_table(f"Mean Metrics", all_metrics_new, logger=logger)
    all_metrics_new.update(all_metrics)
    print('Finished testing on ckpt: ', cfg.TEST.CHECKPOINTS)


if __name__ == "__main__":
    main()
