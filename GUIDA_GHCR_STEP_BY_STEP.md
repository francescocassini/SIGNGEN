# Guida Step-by-Step: Build Docker Full-Deps, Push su GHCR, Run Locale

Questa guida produce e usa una singola immagine Docker con `deps/` (SMPL inclusi) dentro.
Le variabili runtime restano fuori dall'immagine e si cambiano senza rebuild.

## 0) Prerequisiti

- Docker + Docker Compose installati
- GPU NVIDIA + NVIDIA Container Toolkit (se vuoi usare GPU)
- Token GitHub (`GH_TOKEN`) con scope:
  - `write:packages`
  - `read:packages`
- (Opzionale) token Hugging Face (`HF_TOKEN`) e Weights & Biases (`WANDB_API_KEY`)

## 1) Vai nella repo

```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
```

## 2) Prepara file runtime env (non incluso nell'immagine)

### Opzione A (consigliata)

```bash
scripts/init_docker_env.sh
# crea .env.runtime
```

### Opzione B manuale

```bash
cp .env.runtime.example .env.runtime
```

Apri `.env.runtime` e imposta almeno:
- `HF_TOKEN`
- `WANDB_API_KEY` (se usi wandb)
- `SOKE_HF_DATASET_REPO` (se vuoi auto-download dataset)
- impostazioni GPU (`SOKE_TRAIN_USE_GPUS`, `SOKE_TRAIN_DEVICE_IDS`, ecc.)

## 3) Build immagine full-deps

```bash
docker compose build --no-cache soke
```

L'immagine locale risultante e':
- `signgen/soke:local`

## 4) Verifica rapida immagine locale

```bash
docker images | grep 'signgen/soke'
```

GPU check:

```bash
docker compose run --rm soke nvidia-smi
```

## 5) Login GHCR

```bash
echo "<GH_TOKEN>" | docker login ghcr.io -u francescocassini --password-stdin
```

## 6) Tag immagine per GHCR

```bash
docker tag signgen/soke:local ghcr.io/francescocassini/soke:full-deps
```

Opzionale tag latest:

```bash
docker tag signgen/soke:local ghcr.io/francescocassini/soke:latest
```

## 7) Push su GHCR

```bash
docker push ghcr.io/francescocassini/soke:full-deps
# opzionale
# docker push ghcr.io/francescocassini/soke:latest
```

## 8) Pull su qualsiasi macchina target

```bash
echo "<GH_TOKEN>" | docker login ghcr.io -u francescocassini --password-stdin
docker pull ghcr.io/francescocassini/soke:full-deps
```

## 9) Avvio training sul tuo PC

### 9A) Metodo consigliato (script)

```bash
SOKE_DOCKER_IMAGE=ghcr.io/francescocassini/soke:full-deps scripts/run_docker_image.sh train
```

Lo script usa `.env.runtime` e i mount host:
- `${SOKE_DATA_ROOT_HOST}` -> `/workspace/SOKE_DATA`
- `${SOKE_ARTIFACTS_ROOT_HOST}` -> `/workspace/SOKE_ARTIFACTS`
- cache HF host -> `/workspace/.cache/huggingface`

### 9B) Metodo diretto (`docker run`)

```bash
docker run --gpus all --rm -it \
  --env-file /home/cirillo/Desktop/SIGNGEN/SOKE/.env.runtime \
  -v /home/cirillo/Desktop/SOKE_DATA:/workspace/SOKE_DATA \
  -v /home/cirillo/Desktop/SOKE_ARTIFACTS:/workspace/SOKE_ARTIFACTS \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  ghcr.io/francescocassini/soke:full-deps train
```

## 10) Come passare variabili da comando (senza rebuild)

Puoi sovrascrivere qualsiasi variabile runtime con `-e`:

```bash
docker run --gpus all --rm -it \
  --env-file /home/cirillo/Desktop/SIGNGEN/SOKE/.env.runtime \
  -e SOKE_TRAIN_CFG=configs/soke.yaml \
  -e SOKE_TRAIN_USE_GPUS=0,1,2,3 \
  -e SOKE_TRAIN_DEVICE_IDS=0,1,2,3 \
  -e SOKE_TRAIN_BATCH_SIZE=16 \
  -v /home/cirillo/Desktop/SOKE_DATA:/workspace/SOKE_DATA \
  -v /home/cirillo/Desktop/SOKE_ARTIFACTS:/workspace/SOKE_ARTIFACTS \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  ghcr.io/francescocassini/soke:full-deps train
```

## 11) Avvio inferenza/test

```bash
# infer
SOKE_DOCKER_IMAGE=ghcr.io/francescocassini/soke:full-deps scripts/run_docker_image.sh infer

# cycle train->infer
SOKE_DOCKER_IMAGE=ghcr.io/francescocassini/soke:full-deps scripts/run_docker_image.sh cycle
```

## 12) Troubleshooting rapido

- Errore auth GHCR: rifai login e verifica scope token (`write:packages`, `read:packages`).
- Errore GPU: controlla NVIDIA toolkit e prova `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- Dataset mancante: imposta `SOKE_HF_DATASET_REPO` + `HF_TOKEN` in `.env.runtime`.
- Cambio variabili: modifica `.env.runtime` o passa `-e`; non serve rebuild.

## Nota operativa: training originale con GPU diverse (senza modificare il codice)
Per mantenere il codice/config identici all'upstream, non editare `configs/soke.yaml`.
Usa i parametri CLI al lancio:

- 1 GPU (GPU 0):
```bash
python -m train --cfg configs/soke.yaml --nodebug --use_gpus 0 --device 0
```

- 2 GPU (GPU 0,1):
```bash
python -m train --cfg configs/soke.yaml --nodebug --use_gpus 0,1 --device 0 1
```

- 4 GPU (GPU 0,1,2,3):
```bash
python -m train --cfg configs/soke.yaml --nodebug --use_gpus 0,1,2,3 --device 0 1 2 3
```

- 8 GPU (GPU 0..7):
```bash
python -m train --cfg configs/soke.yaml --nodebug --use_gpus 0,1,2,3,4,5,6,7 --device 0 1 2 3 4 5 6 7
```

Nota: cambiare numero di GPU cambia il batch globale effettivo (`batch_per_gpu * num_gpu`), quindi il run non e' matematicamente identico se non si mantiene invariato anche il batch globale.
