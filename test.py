import json
import os
import time
import subprocess
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pathlib import Path
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger
from mGPT.callback import build_callbacks
from mGPT.config import parse_args, instantiate_from_config
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


def _env_bool(name, default=False):
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "on"}


def _iter_loggers(trainer):
    if getattr(trainer, "loggers", None):
        return list(trainer.loggers)
    if getattr(trainer, "logger", None) is not None:
        return [trainer.logger]
    return []


def log_test_summary_to_wandb(trainer, mean_metrics, replication_times, logger):
    if not mean_metrics:
        return
    payload = {
        f"test_summary/{k}": float(v)
        for k, v in mean_metrics.items()
    }
    payload["test_summary/replication_times"] = int(replication_times)
    payload["test_summary/timestamp"] = int(time.time())
    sent = False
    for pl_logger in _iter_loggers(trainer):
        name = pl_logger.__class__.__name__.lower()
        if "wandb" not in name:
            continue
        try:
            experiment = getattr(pl_logger, "experiment", None)
            if experiment is not None:
                experiment.log(payload)
                sent = True
        except Exception as e:
            logger.warning("Unable to log test summary to W&B: %s", e)
    if sent:
        logger.info("Test summary metrics sent to W&B (%d items).", len(mean_metrics))


def _build_quality_metrics_message(cfg, mean_metrics):
    key_order = [
        "Metrics/how2sign_DTW_MPJPE_PA_lhand/mean",
        "Metrics/how2sign_DTW_MPJPE_PA_rhand/mean",
        "Metrics/how2sign_DTW_MPJPE_PA_body/mean",
        "Metrics/csl_DTW_MPJPE_PA_lhand/mean",
        "Metrics/csl_DTW_MPJPE_PA_rhand/mean",
        "Metrics/csl_DTW_MPJPE_PA_body/mean",
        "Metrics/phoenix_DTW_MPJPE_PA_lhand/mean",
        "Metrics/phoenix_DTW_MPJPE_PA_rhand/mean",
        "Metrics/phoenix_DTW_MPJPE_PA_body/mean",
        "Metrics/avg_DTW_MPJPE_PA_lhand/mean",
        "Metrics/avg_DTW_MPJPE_PA_rhand/mean",
        "Metrics/avg_DTW_MPJPE_PA_body/mean",
        "Metrics/avg_DTW_MPJPE_PA_hand/mean",
        "Metrics/avg_DTW_MPJPE_PA_hand_body/mean",
    ]
    labels = {
        "Metrics/how2sign_DTW_MPJPE_PA_lhand/mean": "how2sign lhand",
        "Metrics/how2sign_DTW_MPJPE_PA_rhand/mean": "how2sign rhand",
        "Metrics/how2sign_DTW_MPJPE_PA_body/mean": "how2sign body",
        "Metrics/csl_DTW_MPJPE_PA_lhand/mean": "csl lhand",
        "Metrics/csl_DTW_MPJPE_PA_rhand/mean": "csl rhand",
        "Metrics/csl_DTW_MPJPE_PA_body/mean": "csl body",
        "Metrics/phoenix_DTW_MPJPE_PA_lhand/mean": "phoenix lhand",
        "Metrics/phoenix_DTW_MPJPE_PA_rhand/mean": "phoenix rhand",
        "Metrics/phoenix_DTW_MPJPE_PA_body/mean": "phoenix body",
        "Metrics/avg_DTW_MPJPE_PA_lhand/mean": "avg lhand",
        "Metrics/avg_DTW_MPJPE_PA_rhand/mean": "avg rhand",
        "Metrics/avg_DTW_MPJPE_PA_body/mean": "avg body",
        "Metrics/avg_DTW_MPJPE_PA_hand/mean": "avg hand",
        "Metrics/avg_DTW_MPJPE_PA_hand_body/mean": "avg hand+body",
    }

    lines = [
        f"[SOKE][test] QUALITY SUMMARY name={cfg.NAME} ckpt={Path(cfg.TEST.CHECKPOINTS).name}",
        "DTW_MPJPE_PA metrics (hand/body):",
    ]
    found = 0
    for key in key_order:
        if key in mean_metrics:
            lines.append(f"- {labels[key]}: {float(mean_metrics[key]):.3f}")
            found += 1
    if found == 0:
        lines.append("- no quality metrics found")
    return "\n".join(lines)


def send_quality_metrics_telegram(cfg, mean_metrics, logger):
    if not _env_bool("SOKE_TELEGRAM_NOTIFY", True):
        return
    root_dir = Path(__file__).resolve().parent
    script = root_dir / "scripts" / "telegram_notify.sh"
    if not script.is_file():
        logger.warning("Telegram script not found: %s", script)
        return
    message = _build_quality_metrics_message(cfg, mean_metrics)
    try:
        subprocess.run([str(script), "text", message], check=False)
        logger.info("Quality summary sent to Telegram.")
    except Exception as e:
        logger.warning("Unable to send quality summary to Telegram: %s", e)


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.USE_GPUS
    cfg.FOLDER = cfg.TEST.FOLDER
    if not os.environ.get("WANDB_API_KEY"):
        cfg.LOGGER.WANDB.params.offline = True
        os.environ["WANDB_MODE"] = "disabled"
        cfg.LOGGER.TYPE = [x for x in cfg.LOGGER.TYPE if x.lower() != "wandb"]
        if not cfg.LOGGER.TYPE:
            cfg.LOGGER.TYPE = ["tensorboard"]

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

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName.lower() in {'tensorboard', 'wandb'}:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)
    csv_logger = CSVLogger(save_dir=cfg.FOLDER_EXP, name="csv_logs", version="")
    pl_loggers.append(csv_logger)
    logger.info(f"CSV metrics log: {csv_logger.log_dir}")

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
        logger=pl_loggers,
        callbacks=callbacks,
    )

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # loading state dict
    if not cfg.TEST.CHECKPOINTS:
        candidates = []
        # Primary path derived from the current test run name.
        candidates.append(
            os.path.join(
                cfg.FOLDER_EXP.replace("results", "experiments"),
                "checkpoints",
                "last.ckpt",
            )
        )
        # Common training run name used in this project.
        candidates.append(
            os.path.join(
                cfg.FOLDER.replace("results", "experiments"),
                "mgpt",
                "SOKE",
                "checkpoints",
                "last.ckpt",
            )
        )
        # Optional explicit default from env for production/server setups.
        env_ckpt = os.environ.get("SOKE_DEFAULT_TEST_CKPT", "").strip()
        if env_ckpt:
            candidates.insert(0, env_ckpt)

        chosen = next((p for p in candidates if p and os.path.isfile(p)), None)
        if chosen is None:
            pretty = "\n".join([f"  - {p}" for p in candidates if p])
            raise FileNotFoundError(
                "No inference checkpoint found. Searched:\n"
                f"{pretty}\n"
                "Provide one with `--checkpoint <path>` or set SOKE_DEFAULT_TEST_CKPT."
            )
        cfg.TEST.CHECKPOINTS = chosen
        logger.info("Auto-selected checkpoint for inference: %s", chosen)
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

    mean_metrics = {}
    all_metrics_new = {}

    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item),
                                                    replication_times)
        metric_key = key + "/mean"
        mean_metrics[metric_key] = float(mean)
        all_metrics_new[metric_key] = f"{mean:.3f}"
        # all_metrics_new[key + "/conf_interval"] = conf_interval

    log_test_summary_to_wandb(
        trainer=trainer,
        mean_metrics=mean_metrics,
        replication_times=replication_times,
        logger=logger,
    )
    send_quality_metrics_telegram(cfg=cfg, mean_metrics=mean_metrics, logger=logger)

    print_table(f"Mean Metrics", all_metrics_new, logger=logger)
    all_metrics_new.update(all_metrics)
    print('Finished testing on ckpt: ', cfg.TEST.CHECKPOINTS)


if __name__ == "__main__":
    main()
