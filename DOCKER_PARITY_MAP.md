# SOKE Local-to-Docker Parity Map

Questa guida mappa esattamente il workflow locale che usi oggi verso i comandi Docker equivalenti.

## 1) Prerequisiti minimi (uguali a locale)
- GPU NVIDIA visibile da Docker (`nvidia-smi`).
- Repo codice montata nel container su `/workspace/SOKE`.
- Dataset root montata su `/workspace/SOKE_DATA`.
- Accesso HF privato via env:
  - `SOKE_HF_DATASET_REPO=<USER>/<DATASET_REPO>`
  - `HF_TOKEN=<token>`

Comando rapido di check:

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
docker compose run --rm soke nvidia-smi
docker compose run --rm soke bash -lc "./scripts/check_soke_setup.sh || true"
```

## 2) Training
### Locale (riferimento)

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
source scripts/set_data_root_env.sh /home/cirillo/Desktop/SOKE_DATA
script -q -f logs/train_tty.log -c "PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -u -m train --cfg configs/soke.yaml --nodebug --use_gpus 0 --device 0"
```

### Docker equivalente

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
docker compose run --rm soke train
```

Note:
- L'entrypoint Docker imposta automaticamente `SOKE_DATA_ROOT=/workspace/SOKE_DATA`.
- Se `SOKE_HF_DATASET_REPO` e `HF_TOKEN` sono presenti, fa bootstrap dataset da HF prima del train.

## 3) Monitoring training
### Locale (riferimento)

```bash
watch -n 5 'ls -lah /home/cirillo/Desktop/SIGNGEN/SOKE/experiments/mgpt/SOKE/checkpoints'
find /home/cirillo/Desktop/SIGNGEN/SOKE/experiments/mgpt/SOKE -name metrics.csv
tail -f /home/cirillo/Desktop/SIGNGEN/SOKE/experiments/mgpt/SOKE/csv_logs/metrics.csv
watch -n 1 "nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used,temperature.gpu --format=csv,noheader"
```

### Docker equivalente (host-side, stessi file montati)

```bash
watch -n 5 'ls -lah /home/cirillo/Desktop/SIGNGEN/SOKE/experiments/mgpt/SOKE/checkpoints'
find /home/cirillo/Desktop/SIGNGEN/SOKE/experiments/mgpt/SOKE -name metrics.csv
tail -f /home/cirillo/Desktop/SIGNGEN/SOKE/experiments/mgpt/SOKE/csv_logs/metrics.csv
watch -n 1 "nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used,temperature.gpu --format=csv,noheader"
```

I path sono identici perche' `./` viene bind-mountata nel container.

## 4) Inferenza completa (checkpoint AMG)
### Locale (riferimento)

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
bash scripts/run_inference_complete.sh last
```

### Docker equivalente

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
docker compose run --rm soke bash -lc "bash scripts/run_inference_complete.sh last"
```

Per confronto multi-checkpoint:

```bash
docker compose run --rm soke bash -lc "bash scripts/run_inference_complete.sh all"
```

Preview rapida:

```bash
docker compose run --rm soke bash -lc "MAX_SAMPLES=32 SKIP_METRICS=1 bash scripts/run_inference_complete.sh last"
```

## 5) GIF Ground Truth vs Predicted
Prerequisito: inferenza con `TEST.SAVE_PREDICTIONS=True` (gia' attivo in `configs/soke_infer_complete.yaml`).

### Locale o Docker (stesso comando, path condivisi)

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
python scripts/preview_test_sample.py \
  --pred_dir results/mgpt/SOKE_INFER/test_rank_0 \
  --index 0 \
  --out_dir visualize/preview_samples \
  --fps 20
```

Output attesi:
- `*_pred.gif` (pred-only)
- `*_compare_ref_pred.gif` (GT vs PRED side-by-side)
- `*_text.txt`

## 6) Artefatti da verificare (done criteria)
- Training:
  - `experiments/mgpt/SOKE/checkpoints/*.ckpt`
  - `experiments/mgpt/SOKE/csv_logs/metrics.csv`
- Inferenza:
  - `results/mgpt/SOKE_INFER/test_rank_0/test_scores.json`
  - `results/mgpt/SOKE_INFER/test_rank_0/*.pkl`
- GIF preview:
  - `visualize/preview_samples/*_compare_ref_pred.gif`

## 7) Variabili/env realmente necessarie in Docker
- `SOKE_HF_DATASET_REPO`
- `HF_TOKEN` (solo se repo privata)
- `SOKE_AUTO_DOWNLOAD_DATASET=1`
- `SOKE_TRAIN_CFG=configs/soke.yaml`
- `SOKE_TEST_CFG=configs/soke_infer_complete.yaml`
- `SOKE_DATA_ROOT_HOST=/path/host/SOKE_DATA` (nel compose)

## 8) Nota sicurezza
- Non mettere token in file `.md` o in comandi salvati in chiaro.
- Passare `HF_TOKEN` via `.env` locale git-ignored o tramite env runtime.
