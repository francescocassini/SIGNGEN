import os
import time
import subprocess
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint


def build_callbacks(cfg, logger=None, phase='test', **kwargs):
    callbacks = []
    logger = logger

    # Rich Progress Bar
    callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))
        periodic_cb = PeriodicInferenceCallback(cfg, logger=logger)
        if periodic_cb.enabled:
            callbacks.append(periodic_cb)
        
    return callbacks

def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []
    # Logging
    metric_monitor = {
        "loss_total": "total/train",
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",
        "how2sign_DTW_MPJPE_PA_lhand": "Metrics/how2sign_DTW_MPJPE_PA_lhand",
        "how2sign_DTW_MPJPE_PA_rhand": "Metrics/how2sign_DTW_MPJPE_PA_rhand",
        "how2sign_DTW_MPJPE_PA_body": "Metrics/how2sign_DTW_MPJPE_PA_body",
        "csl_DTW_MPJPE_PA_lhand": "Metrics/csl_DTW_MPJPE_PA_lhand",
        "csl_DTW_MPJPE_PA_rhand": "Metrics/csl_DTW_MPJPE_PA_rhand",
        "csl_DTW_MPJPE_PA_body": "Metrics/csl_DTW_MPJPE_PA_body",
        "phoenix_DTW_MPJPE_PA_lhand": "Metrics/phoenix_DTW_MPJPE_PA_lhand",
        "phoenix_DTW_MPJPE_PA_rhand": "Metrics/phoenix_DTW_MPJPE_PA_rhand",
        "phoenix_DTW_MPJPE_PA_body": "Metrics/phoenix_DTW_MPJPE_PA_body",
        "how2sign_MPVPE_PA_all": "Metrics/how2sign_MPVPE_PA_all",
        "how2sign_MPJPE_PA_hand": "Metrics/how2sign_MPJPE_PA_hand",
        "csl_MPVPE_PA_all": "Metrics/csl_MPVPE_PA_all",
        "csl_MPJPE_PA_hand": "Metrics/csl_MPJPE_PA_hand",
        "phoenix_MPVPE_PA_all": "Metrics/phoenix_MPVPE_PA_all",
        "phoenix_MPJPE_PA_hand": "Metrics/phoenix_MPJPE_PA_hand",
        "BLEU_1": "Metrics/Bleu_1",
        "BLEU_2": "Metrics/Bleu_2",
        "BLEU_3": "Metrics/Bleu_3",
        "BLEU_4": "Metrics/Bleu_4",
        "ROUGE_L": "Metrics/ROUGE_L",
    }
    callbacks.append(
        progressLogger(logger,metric_monitor=metric_monitor,log_every_n_steps=1))

    # # Save latest checkpoints
    # checkpointParams = {
    #     'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
    #     'filename': "{epoch}",
    #     'monitor': "step",
    #     'mode': "max",
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
    #     'save_top_k': 8,
    #     'save_last': True,
    #     'save_on_train_epoch_end': True
    # }
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    # # Save checkpoint every n*10 epochs
    # checkpointParams.update({
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS * 10,
    #     'save_top_k': -1,
    #     'save_last': False
    # })
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    checkpointParams = {
        'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': None,  #cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
        'save_last': True, #None,
        'save_on_train_epoch_end': False
    }
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    metrics = cfg.METRIC.TYPE
    metric_monitor_map = {
        'TemosMetric': {
            'Metrics/APE_root': {
                'abbr': 'APEroot',
                'mode': 'min'
            },
        },
        'TM2TMetrics': {
            'Metrics/how2sign_DTW_MPJPE_PA_lhand': {
                'abbr': 'how2sign_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/how2sign_DTW_MPJPE_PA_body': {
            #     'abbr': 'how2sign_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # },
            'Metrics/csl_DTW_MPJPE_PA_lhand': {
                'abbr': 'csl_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/csl_DTW_MPJPE_PA_body': {
            #     'abbr': 'csl_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # }
            'Metrics/phoenix_DTW_MPJPE_PA_lhand': {
                'abbr': 'phoenix_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/phoenix_DTW_MPJPE_PA_body': {
            #     'abbr': 'phoenix_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # }
        },
        'M2TMetrics': {
            'Metrics/Bleu_4': {
                'abbr': 'BLEU_4',
                'mode': 'max'
            },
            'Metrics/ROUGE_L': {
                'abbr': 'ROUGE_L',
                'mode': 'max'
            },
        },
        'MRMetrics': {
            'Metrics/how2sign_MPJPE_PA_hand': {
                'abbr': 'how2sign_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/how2sign_MPVPE_PA_all': {
            #     'abbr': 'how2sign_MPVPE_PA_all',
            #     'mode': 'min'
            # },
            'Metrics/csl_MPJPE_PA_hand': {
                'abbr': 'csl_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/csl_MPVPE_PA_all': {
            #     'abbr': 'csl_MPVPE_PA_all',
            #     'mode': 'min'
            # },
            'Metrics/phoenix_MPJPE_PA_hand': {
                'abbr': 'phoenix_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/phoenix_MPVPE_PA_all': {
            #     'abbr': 'phoenix_MPVPE_PA_all',
            #     'mode': 'min'
            # },
        },
        'HUMANACTMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UESTCMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UncondMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            }
        }
    }

    # checkpointParams.update({
    #     'every_n_epochs': None,  #cfg.LOGGER.VAL_EVERY_STEPS,
    #     'save_top_k': 1,
    # })

    for metric in metrics:
        if metric in metric_monitor_map.keys():
            metric_monitors = metric_monitor_map[metric]

            # Delete R3 if training VAE
            if cfg.TRAIN.STAGE == 'vae' and metric == 'TM2TMetrics':
                del metric_monitors['Metrics/R_precision_top_3']

            for metric_monitor in metric_monitors:
                checkpointParams.update({
                    'filename':
                    metric_monitor_map[metric][metric_monitor]['mode']
                    + "-" +
                    metric_monitor_map[metric][metric_monitor]['abbr']
                    + "{epoch}",
                    'monitor':
                    metric_monitor,
                    'mode':
                    metric_monitor_map[metric][metric_monitor]['mode'],
                })
                callbacks.append(
                    ModelCheckpoint(**checkpointParams))
    return callbacks

class progressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1,
                 batch_log_every_n_steps: int = 50):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps
        self.batch_log_every_n_steps = max(1, int(batch_log_every_n_steps))
        self._epoch_start_time = None
        self._train_start_time = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self._train_start_time = time.time()
        total_batches = getattr(trainer, "num_training_batches", "unknown")
        self.logger.info(f"Training started | total_train_batches={total_batches}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        elapsed = 0.0 if self._train_start_time is None else (time.time() - self._train_start_time)
        self.logger.info(f"Training done | total_time {elapsed:.1f}s")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        self._epoch_start_time = time.time()
        total_batches = getattr(trainer, "num_training_batches", "unknown")
        self.logger.info(f"Epoch {trainer.current_epoch} started | train_batches={total_batches}")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, **kwargs) -> None:
        # Frequent plain-text progress logs for non-interactive sessions.
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0 or step % self.batch_log_every_n_steps != 0:
            return

        total_batches = getattr(trainer, "num_training_batches", None)
        if isinstance(total_batches, int) and total_batches > 0:
            pct = 100.0 * float(batch_idx + 1) / float(total_batches)
            batch_progress = f"{batch_idx + 1}/{total_batches} ({pct:.1f}%)"
        else:
            batch_progress = f"{batch_idx + 1}/?"

        elapsed = 0.0 if self._epoch_start_time is None else (time.time() - self._epoch_start_time)
        speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
        if isinstance(total_batches, int) and total_batches > 0 and speed > 0:
            rem = max(total_batches - (batch_idx + 1), 0)
            eta = rem / speed
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "unknown"

        gpu_msg = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                mem_alloc = torch.cuda.memory_allocated(dev) / (1024 ** 2)
                mem_res = torch.cuda.memory_reserved(dev) / (1024 ** 2)
                gpu_msg = f"cuda:{dev} mem_alloc={mem_alloc:.0f}MB mem_reserved={mem_res:.0f}MB"
        except Exception:
            pass

        self.logger.info(
            f"Train progress | epoch={trainer.current_epoch} | step={step} | batch={batch_progress} | "
            f"speed={speed:.2f} batch/s | epoch_elapsed={elapsed:.1f}s | eta={eta_str} | {gpu_msg}"
        )

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")
            return
        # Print monitored validation metrics at each validation epoch.
        metric_format = f"{{:.{self.precision}e}}"
        losses_dict = trainer.callback_metrics
        metrics_str = []
        for metric_name, dico_name in self.metric_monitor.items():
            if dico_name in losses_dict:
                metric = losses_dict[dico_name].item()
                metrics_str.append(f"{metric_name} {metric_format.format(metric)}")
        if metrics_str:
            self.logger.info(
                f"Validation epoch {trainer.current_epoch}: " + "   ".join(metrics_str)
            )

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)
        elapsed = 0.0
        if self._epoch_start_time is not None:
            elapsed = time.time() - self._epoch_start_time
        line = f"{line}   epoch_time {elapsed:.1f}s"
        self.logger.info(line)


class PeriodicInferenceCallback(Callback):
    """Run inference preview every N epochs, then optionally send GIF via existing script hooks."""

    def __init__(self, cfg, logger=None):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.project_root = os.getcwd()
        self.every_n = self._env_int("SOKE_PERIODIC_INFER_EVERY_N_EPOCHS", 0)
        self.max_samples = os.environ.get("SOKE_PERIODIC_INFER_MAX_SAMPLES", "").strip()
        self.skip_metrics = self._env_bool("SOKE_PERIODIC_INFER_SKIP_METRICS", True)
        self.keep_ckpt = self._env_bool("SOKE_PERIODIC_INFER_KEEP_CKPT", False)
        self.enabled = self.every_n > 0

        if self.enabled and self.logger is not None:
            self.logger.info(
                "Periodic inference enabled | every_n_epochs=%s | max_samples=%s | skip_metrics=%s | keep_ckpt=%s",
                self.every_n,
                self.max_samples if self.max_samples else "none",
                self.skip_metrics,
                self.keep_ckpt,
            )

    @staticmethod
    def _env_int(name, default):
        raw = os.environ.get(name, "").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    @staticmethod
    def _env_bool(name, default):
        raw = os.environ.get(name, "").strip().lower()
        if raw == "":
            return default
        return raw in {"1", "true", "yes", "on"}

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled:
            return
        if not getattr(trainer, "is_global_zero", True):
            return

        epoch_done = int(trainer.current_epoch) + 1
        if epoch_done <= 0 or epoch_done % self.every_n != 0:
            return

        ckpt_dir = os.path.join(self.cfg.FOLDER_EXP, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"periodic_epoch={epoch_done}.ckpt")
        trainer.save_checkpoint(ckpt_path)

        env = os.environ.copy()
        if self.max_samples:
            env["MAX_SAMPLES"] = self.max_samples
        if self.skip_metrics:
            env["SKIP_METRICS"] = "1"
        # Keep GIF generation and telegram notifications aligned with existing scripts.
        env.setdefault("SOKE_PREVIEW_GIF_ON_INFER", "1")

        cmd = ["bash", "scripts/run_inference_complete.sh", ckpt_path]
        if self.logger is not None:
            self.logger.info("Periodic inference start | epoch=%s | cmd=%s", epoch_done, " ".join(cmd))

        result = subprocess.run(cmd, cwd=self.project_root, env=env, check=False)
        if result.returncode != 0 and self.logger is not None:
            self.logger.warning(
                "Periodic inference failed | epoch=%s | returncode=%s",
                epoch_done,
                result.returncode,
            )
        elif self.logger is not None:
            self.logger.info("Periodic inference done | epoch=%s", epoch_done)

        if not self.keep_ckpt:
            try:
                os.remove(ckpt_path)
            except OSError:
                pass
