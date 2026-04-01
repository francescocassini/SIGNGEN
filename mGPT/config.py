import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
from os.path import join as pjoin
import os
import glob


def get_module_config(cfg, filepath="./configs"):
    """
    Load yaml config files from subfolders
    """

    yamls = glob.glob(pjoin(filepath, '*', '*.yaml'))
    yamls = [y.replace(filepath, '') for y in yamls]
    for yaml in yamls:
        nodes = yaml.replace('.yaml', '').replace(os.sep, '.')
        nodes = nodes[1:] if nodes[0] == '.' else nodes
        OmegaConf.update(cfg, nodes, OmegaConf.load('./configs' + yaml))

    return cfg


def get_obj_from_str(string, reload=False):
    """
    Get object from string
    """

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate object from config
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def resume_config(cfg: OmegaConf):
    """
    Resume model and wandb
    """
    
    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        if os.path.exists(resume):
            # Checkpoints
            cfg.TRAIN.PRETRAINED = pjoin(resume, "checkpoints", "last.ckpt")
            # Wandb
            wandb_files = os.listdir(pjoin(resume, "wandb", "latest-run"))
            wandb_run = [item for item in wandb_files if "run-" in item][0]
            cfg.LOGGER.WANDB.params.id = wandb_run.replace("run-","").replace(".wandb", "")
        else:
            raise ValueError("Resume path is not right.")

    return cfg


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name, default=False):
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "on"}


def apply_env_overrides(cfg: OmegaConf, phase: str):
    # Training controls for quick local/remote tests.
    cfg.TRAIN.END_EPOCH = _env_int("SOKE_TRAIN_END_EPOCH", cfg.TRAIN.END_EPOCH)
    cfg.LOGGER.VAL_EVERY_STEPS = max(1, _env_int("SOKE_VAL_EVERY_EPOCHS", cfg.LOGGER.VAL_EVERY_STEPS))
    cfg.TRAIN.BATCH_SIZE = _env_int("SOKE_TRAIN_BATCH_SIZE", cfg.TRAIN.BATCH_SIZE)
    cfg.TEST.BATCH_SIZE = _env_int("SOKE_TEST_BATCH_SIZE", cfg.TEST.BATCH_SIZE)

    # Test-side controls (useful both for explicit infer runs and periodic previews).
    max_samples = os.environ.get("SOKE_TEST_MAX_SAMPLES", "").strip()
    if max_samples != "":
        cfg.TEST.MAX_SAMPLES = int(max_samples)
    if _env_bool("SOKE_TEST_SKIP_METRICS", False):
        cfg.TEST.SKIP_METRICS = True

    if phase == "test":
        default_ckpt = os.environ.get("SOKE_DEFAULT_TEST_CKPT", "").strip()
        if default_ckpt and not cfg.TEST.CHECKPOINTS:
            cfg.TEST.CHECKPOINTS = default_ckpt

    return cfg

def parse_args(phase="train"):
    """
    Parse arguments and load config files
    """

    parser = ArgumentParser()
    group = parser.add_argument_group("Training options")

    # Assets
    group.add_argument(
        "--cfg_assets",
        type=str,
        required=False,
        default="./configs/assets.yaml",
        help="config file for asset paths",
    )

    # Default config
    if phase in ["train", "test", "demo"]:
        cfg_defualt = "./configs/default.yaml"
    elif phase == "render":
        cfg_defualt = "./configs/render.yaml"
    elif phase == "webui":
        cfg_defualt = "./configs/webui.yaml"
        
    group.add_argument(
        "--cfg",
        type=str,
        required=False,
        default=cfg_defualt,
        help="config file",
    )
    group.add_argument("--use_gpus",
                           type=str,
                           required=False,
                           default='0',
                           help="cuda environ devices")
    
    # Parse for each phase
    if phase in ["train", "test"]:
        group.add_argument("--batch_size",
                           type=int,
                           required=False,
                           help="training batch size")
        group.add_argument("--num_nodes",
                           type=int,
                           required=False,
                           help="number of nodes")
        group.add_argument("--device",
                           type=int,
                           nargs="+",
                           required=False,
                           help="training device")
        group.add_argument("--task",
                           type=str,
                           required=False,
                           help="evaluation task type")
        group.add_argument("--nodebug",
                           action="store_true",
                           required=False,
                           help="debug or not")
        group.add_argument("--checkpoint",
                           type=str,
                           required=False,
                           help="checkpoint path for test/inference")
        group.add_argument("--test_max_samples",
                           type=int,
                           required=False,
                           help="limit number of TEST samples (preview mode)")
        group.add_argument("--skip_metrics",
                           action="store_true",
                           required=False,
                           help="skip expensive test metrics (faster preview)")


    if phase == "demo":
        group.add_argument("--task",
            type=str,
            required=False,
            help="evaluation task type")
        group.add_argument(
            "--example",
            type=str,
            required=False,
            help="input text and lengths with txt format",
        )
        group.add_argument(
            "--out_dir",
            type=str,
            required=False,
            help="output dir",
        )
        group.add_argument(
            "--demo_dataset",
            default=None,
            type=str,
            required=False,
            help="output dir",
        )

    if phase == "render":
        group.add_argument("--npy",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion files")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion folder")
        group.add_argument("--fps",
                    type=int,
                    required=False,
                    default=30,
                    help="render fps")
        group.add_argument(
            "--mode",
            type=str,
            required=False,
            default="sequence",
            help="render target: video, sequence, frame",
        )

    params = parser.parse_args()
    
    # Load yaml config files
    OmegaConf.register_new_resolver("eval", eval)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg_base = OmegaConf.load(pjoin(cfg_assets.CONFIG_FOLDER, 'default.yaml'))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    cfg.USE_GPUS = params.use_gpus
    # Update config with arguments
    if phase in ["train", "test"]:
        cfg.TRAIN.BATCH_SIZE = params.batch_size if params.batch_size else cfg.TRAIN.BATCH_SIZE
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.NUM_NODES = params.num_nodes if params.num_nodes else cfg.NUM_NODES
        cfg.model.params.task = params.task if params.task else cfg.model.params.task
        cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG
        if phase == "test" and params.checkpoint:
            cfg.TEST.CHECKPOINTS = params.checkpoint
        if phase == "test" and params.test_max_samples is not None:
            cfg.TEST.MAX_SAMPLES = int(params.test_max_samples)
        if phase == "test" and params.skip_metrics:
            cfg.TEST.SKIP_METRICS = True

        # Force no debug in test
        if phase == "test":
            cfg.DEBUG = False
            # cfg.DEVICE = [0]
            print("Force no debugging when testing")

    if phase == "demo":
        cfg.DEMO_DATASET = params.demo_dataset
        cfg.DEMO.EXAMPLE = params.example
        cfg.DEMO.TASK = params.task
        cfg.TEST.FOLDER = params.out_dir if params.out_dir else cfg.TEST.FOLDER
        os.makedirs(cfg.TEST.FOLDER, exist_ok=True)

    if phase == "render":
        if params.npy:
            cfg.RENDER.NPY = params.npy
            cfg.RENDER.INPUT_MODE = "npy"
        if params.dir:
            cfg.RENDER.DIR = params.dir
            cfg.RENDER.INPUT_MODE = "dir"
        if params.fps:
            cfg.RENDER.FPS = float(params.fps)
        cfg.RENDER.MODE = params.mode

    # Debug mode
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        cfg.LOGGER.WANDB.params.offline = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1

    # Optional env-driven runtime overrides (docker/.env friendly).
    cfg = apply_env_overrides(cfg, phase)
        
    # Resume config
    cfg = resume_config(cfg)

    return cfg
