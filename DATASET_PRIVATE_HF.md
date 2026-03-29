# SOKE Dataset As Private Hugging Face Repo

This guide keeps `SOKE` code repo light and stores dataset files in a **separate private repo** on Hugging Face.

## 1) Target local layout (external to SOKE repo)
Recommended folder on Desktop:

```bash
/home/cirillo/Desktop/SOKE_DATA
```

Expected structure used by SOKE:

```text
SOKE_DATA/
  How2Sign/
    train/
      poses/
      re_aligned/how2sign_realigned_train_preprocessed_fps.csv
    val/
      poses/
      re_aligned/how2sign_realigned_val_preprocessed_fps.csv
    test/
      poses/
      re_aligned/how2sign_realigned_test_preprocessed_fps.csv
    TOKENS_h2s_csl_phoenix/   # motion tokens (if generated)
  CSL-Daily/
    poses/
    csl_clean.train
    csl_clean.val
    csl_clean.test
    mean.pt
    std.pt
  Phoenix_2014T/
    phoenix14t.train
    phoenix14t.dev
    phoenix14t.test
    <utterance_folders>/
```

## 2) Create private dataset repo on Hugging Face
Replace `<USER>` and `<DATASET_REPO>`:

```bash
huggingface-cli login
huggingface-cli repo create <DATASET_REPO> --type dataset --private
```

Then push your existing local dataset folder:

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
scripts/hf_dataset_push_private.sh <USER>/<DATASET_REPO> /home/cirillo/Desktop/SOKE_DATA
```

## 3) Enable auto-download when dataset is missing
SOKE now auto-checks required dataset files at startup (train/test).  
If files are missing, it tries to clone/pull from your private HF dataset repo.

Set:

```bash
export SOKE_HF_DATASET_REPO=<USER>/<DATASET_REPO>
```

For private access in Docker/servers, pass token as env (do not hardcode in repo):

```bash
export HF_TOKEN=<your_hf_token>
```

Recommended (same shell session):

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
source scripts/set_data_root_env.sh /home/cirillo/Desktop/SOKE_DATA
export SOKE_HF_DATASET_REPO=<USER>/<DATASET_REPO>
```

If `/home/cirillo/Desktop/SOKE_DATA` is missing/incomplete, SOKE will sync it automatically from HF.

You can also force the download as an explicit first step:

```bash
scripts/download_dataset_from_hf.sh <USER>/<DATASET_REPO> /home/cirillo/Desktop/SOKE_DATA
```

## 4) Use external dataset with SOKE
From SOKE repo:

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
source scripts/set_data_root_env.sh /home/cirillo/Desktop/SOKE_DATA
```

Validate:

```bash
./scripts/check_soke_setup.sh
```

## 5) New machine workflow
After `git pull` on SOKE code repo:

```bash
huggingface-cli login
cd /path/to/SOKE
source scripts/set_data_root_env.sh /home/cirillo/Desktop/SOKE_DATA
export SOKE_HF_DATASET_REPO=<USER>/<DATASET_REPO>
./scripts/check_soke_setup.sh
```

You can skip manual `git clone` because auto-download handles first sync.

## 6) Notes
- Keep dataset repo private if data licenses require restricted access.
- Do not commit `SOKE_DATA` inside the SOKE code repo.
- `deps/` and `experiments/` are runtime/model artifacts and are already excluded from git in this repo.
