import os
import subprocess
from pathlib import Path
import tarfile
import zipfile


def _log(msg: str):
    print(f"[dataset-autodl] {msg}")


def _required_paths(cfg):
    h2s_root = Path(cfg.DATASET.H2S.ROOT)
    csl_root = Path(cfg.DATASET.H2S.CSL_ROOT)
    pho_root = Path(cfg.DATASET.H2S.PHOENIX_ROOT)
    mean_path = Path(cfg.DATASET.H2S.MEAN_PATH)
    std_path = Path(cfg.DATASET.H2S.STD_PATH)

    req = [
        h2s_root / "train" / "re_aligned" / "how2sign_realigned_train_preprocessed_fps.csv",
        h2s_root / "val" / "re_aligned" / "how2sign_realigned_val_preprocessed_fps.csv",
        h2s_root / "test" / "re_aligned" / "how2sign_realigned_test_preprocessed_fps.csv",
        csl_root / "csl_clean.train",
        csl_root / "csl_clean.val",
        csl_root / "csl_clean.test",
        pho_root / "phoenix14t.train",
        pho_root / "phoenix14t.dev",
        pho_root / "phoenix14t.test",
        mean_path,
        std_path,
    ]
    return req


def _all_present(paths):
    return all(p.exists() for p in paths)


def _resolve_data_root(cfg):
    if os.environ.get("SOKE_DATA_ROOT"):
        return Path(os.environ["SOKE_DATA_ROOT"]).expanduser().resolve()
    # Fallback: infer as common parent for dataset roots.
    h2s_root = Path(cfg.DATASET.H2S.ROOT).resolve()
    return h2s_root.parent


def _run(cmd, cwd=None, check=True):
    _log("run: " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check)


def _extract_archive(archive_path: Path, target_root: Path):
    _log(f"extracting archive: {archive_path.name}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_root)
        return

    # tar, tar.gz, tgz
    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(target_root)


def _extract_known_archives_if_present(data_root: Path):
    # Archive-first mode: upload a few big files instead of millions of tiny files.
    archive_names = [
        "How2Sign.tar.gz",
        "CSL-Daily.tar.gz",
        "Phoenix_2014T.tar.gz",
        "How2Sign.zip",
        "CSL-Daily.zip",
        "Phoenix_2014T.zip",
    ]
    found = False
    for name in archive_names:
        archive_path = data_root / name
        if archive_path.exists():
            found = True
            _extract_archive(archive_path, data_root)
    if found:
        _log("archive extraction completed")


def _sync_with_hf_hub(repo_id: str, data_root: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        _log(f"huggingface_hub not available ({e}), fallback to git/lfs")
        return False

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    _log("trying snapshot download via huggingface_hub")
    data_root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_root),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    return True


def ensure_dataset_available(cfg):
    req = _required_paths(cfg)
    if _all_present(req):
        return

    repo_id = os.environ.get("SOKE_HF_DATASET_REPO", "").strip()
    if not repo_id:
        missing = [str(p) for p in req if not p.exists()][:5]
        raise FileNotFoundError(
            "Dataset files are missing and SOKE_HF_DATASET_REPO is not set.\n"
            "Set SOKE_HF_DATASET_REPO=<user>/<private_dataset_repo> and login via 'huggingface-cli login'.\n"
            f"Example missing paths: {missing}"
        )

    data_root = _resolve_data_root(cfg)
    hf_url = f"https://huggingface.co/datasets/{repo_id}"
    _log(f"dataset missing, trying Hugging Face private dataset: {repo_id}")
    _log(f"target local data root: {data_root}")

    data_root.parent.mkdir(parents=True, exist_ok=True)

    # Preferred: HF Hub API (supports HF_TOKEN without git credentials).
    synced = False
    try:
        synced = _sync_with_hf_hub(repo_id, data_root)
    except Exception as e:
        _log(f"snapshot download failed ({e}), fallback to git/lfs")

    # Fallback: git-lfs clone/pull.
    if not synced:
        _run(["git", "lfs", "install"], check=False)

        if not data_root.exists():
            _run(["git", "clone", hf_url, str(data_root)])
        elif (data_root / ".git").exists():
            _run(["git", "-C", str(data_root), "pull"], check=False)
        else:
            raise RuntimeError(
                f"Data root exists but is not a git repo: {data_root}\n"
                "Please remove/rename it, or set SOKE_DATA_ROOT to a clean path."
            )

        if (data_root / ".git").exists():
            _run(["git", "-C", str(data_root), "lfs", "pull"], check=False)

    # Re-check
    if not _all_present(req):
        _extract_known_archives_if_present(data_root)

    req_after = _required_paths(cfg)
    if not _all_present(req_after):
        missing = [str(p) for p in req_after if not p.exists()][:10]
        raise FileNotFoundError(
            "Hugging Face dataset sync finished but required files are still missing.\n"
            f"Missing examples: {missing}"
        )

    _log("dataset ready")
