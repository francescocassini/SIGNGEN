import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf
from mGPT.callback import build_callbacks
from mGPT.config import parse_args, instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.USE_GPUS
    if not os.environ.get("WANDB_API_KEY"):
        cfg.LOGGER.WANDB.params.offline = True
        os.environ["WANDB_MODE"] = "disabled"
        # In restricted environments, wandb service can fail even in offline mode.
        cfg.LOGGER.TYPE = [x for x in cfg.LOGGER.TYPE if x.lower() != "wandb"]
        if not cfg.LOGGER.TYPE:
            cfg.LOGGER.TYPE = ["tensorboard"]
    # print(cfg)

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(OmegaConf.to_yaml(cfg))  # print config file

    if cfg.ACCELERATOR == "gpu" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        cfg.ACCELERATOR = "cpu"
        cfg.DEVICE = 1
        cfg.NUM_NODES = 1
        # Multiprocessing workers can fail in restricted CPU sandboxes.
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
    # Always keep a plain CSV logger for easy CLI monitoring.
    csv_logger = CSVLogger(save_dir=cfg.FOLDER_EXP, name="csv_logs", version="")
    pl_loggers.append(csv_logger)
    logger.info(f"CSV metrics log: {csv_logger.log_dir}")

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))
    if len(datamodule.train_dataset) == 0:
        raise RuntimeError(
            "No training samples found. Populate pose directories under "
            "../data/How2Sign, ../data/CSL-Daily/poses, ../data/Phoenix_2014T."
        )

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        # precision='16',
        logger=pl_loggers,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true" if isinstance(cfg.DEVICE, (list, tuple)) and len(cfg.DEVICE) > 1 else 'auto',
        # strategy=DDPStrategy(process_group_backend="nccl"),
        benchmark=False,
        deterministic=False,
        # num_sanity_val_steps=0,  #for debug
    )
    logger.info("Trainer initialized")

    # Strict load pretrianed model
    if cfg.TRAIN.PRETRAINED:
        load_pretrained(cfg, model, logger)

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # Pytorch 2.0 Compile
    # if torch.__version__ >= "2.0.0":
    #     model = torch.compile(model, mode="reduce-overhead")
    # model = torch.compile(model)

    print('tmax: ', cfg.TRAIN.LR_SCHEDULER.params.T_max)
    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
