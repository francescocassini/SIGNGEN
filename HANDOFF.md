# HANDOFF - SOKE (Signs as Tokens)

Ultimo aggiornamento: 2026-03-29 (Europe/Rome)

## 1) Obiettivo del progetto
Portare la pipeline SOKE in stato **end-to-end funzionante** (train/inferenza/valutazione/visualizzazione), con priorita' immediata su:
- inferenza AMG gia' addestrato (anche non perfetto),
- verifica qualitativa output,
- hardening dei passaggi per riproducibilita'.

## 2) Stato attuale (fact-based)
### Disponibile e pronto
- Checkpoint AMG presenti in `experiments/mgpt/SOKE/checkpoints/`:
  - `last.ckpt`, `last-v1.ckpt`, `last-v2.ckpt`
  - `min-how2sign_DTW_MPJPE_PA_lhandepoch=3.ckpt`
  - `min-csl_DTW_MPJPE_PA_lhandepoch=3.ckpt`
  - `min-phoenix_DTW_MPJPE_PA_lhandepoch=3.ckpt`
- Tokenizer checkpoint disponibile:
  - `deps/tokenizer_ckpt/tokenizer.ckpt`
- mBART disponibile:
  - `deps/mbart-h2s-csl-phoenix/`
- Dataset indicizzati e caricabili in test (how2sign/csl/phoenix):
  - How2Sign usable: 2308
  - CSL usable: 1176
  - Phoenix usable: 642

### Verifiche eseguite oggi
- Smoke test inferenza (`python -m test --cfg configs/soke.yaml --task t2m`) avviato con successo fino a:
  - caricamento modello AMG,
  - caricamento VAE tokenizer,
  - caricamento checkpoint `last.ckpt`,
  - avvio `trainer.test(...)`.
- Quindi: **si, puoi gia' fare inferenza sui checkpoint presenti**.

## 3) Bug trovati e fix applicati
File: `test.py`
- Fix 1: gestione robusta di `cfg.DEVICE` (OmegaConf ListConfig) per `num_devices`.
  - Prima: crash `TypeError: int() argument must be ... ListConfig`.
- Fix 2: rimosso blocco errato fuori scope dopo `main()` che referenziava `cfg/logger` non definiti.
- Fix 3: aggiunto fallback CPU quando GPU non disponibile (coerente con `train.py`).

Nota: in questo ambiente sandbox la GPU Lightning non e' disponibile; sul tuo ambiente reale con GPU il test procede normalmente.

## 4) Come lanciare inferenza ADESSO
Dalla root `SOKE`:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -u -m test --cfg configs/soke.yaml --task t2m --nodebug --use_gpus 0 --device 0
```

Checkpoint usato di default:
- `experiments/mgpt/SOKE/checkpoints/last.ckpt`

Per output predizioni (se vuoi salvare file generati):
- metti `TEST.SAVE_PREDICTIONS: True` in `configs/soke.yaml`.

Visualizzazione mesh:
```bash
python -m vis_mesh --cfg=configs/soke.yaml --demo_dataset=csl
```

### Pipeline consigliata (nuova)
Script pronto:
```bash
scripts/run_inference_complete.sh last
scripts/run_inference_complete.sh all
```

Config dedicata:
- `configs/soke_infer_complete.yaml` (ha `TEST.SAVE_PREDICTIONS: True`).

Se vuoi forzare un checkpoint specifico:
```bash
python -u -m test --cfg configs/soke_infer_complete.yaml --task t2m --nodebug --use_gpus 0 --device 0 --checkpoint experiments/mgpt/SOKE/checkpoints/last.ckpt
```

Nota tecnica:
- e' stato aggiunto il parametro CLI `--checkpoint` in `mGPT/config.py` per selezionare checkpoint senza editare YAML.

## 5) Lettura performance (stato training attuale)
- Train loss in miglioramento netto (epoche iniziali): ~8.13 -> ~4.61.
- Throughput stabile (~1.6-1.8 it/s, ~19 min/epoca).
- Metriche cross-dataset ancora non affidabili per confronto finale:
  - `csl/phoenix` spesso a 0 nelle prime validazioni,
  - salto anomalo metrica how2sign intorno a epoca 7.

## 6) Gap da chiudere per "SOKE completo"
1. Stabilizzare validazione multi-dataset (no bias sui primi sample).
2. Verificare split/loader per evitare mismatch train/val/test in tutti i dataset wrapper.
3. Definire protocollo inferenza qualitativa (prompt set fisso + render + confronto).
4. Definire protocollo valutazione quantitativa finale (metriche + checkpoint selection).
5. Congelare una config di produzione (`configs/soke_infer.yaml`) per evitare editing manuale continuo.

## 7) Piano operativo (roadmap breve)
### Fase A - Inferenza dimostrativa immediata
- Eseguire test su `last.ckpt` e un `min-*.ckpt`.
- Salvare predizioni e render 5-10 esempi per dataset.
- Output atteso: evidenza visiva che AMG genera sequenze coerenti.

### Fase B - Sanity tecnica pipeline
- Confermare che retrieval token path (`name2kws_*.json`, `word2code.json`) sia usato correttamente.
- Verificare coerenza split e metriche per how2sign/csl/phoenix.
- Output atteso: metriche non degenerate (niente zeri "falsi").

### Fase C - Test completo SOKE
- Run test completo replicabile.
- Report metriche + sample qualitativi.
- Se necessario, resume training AMG e rieseguire test.

## 8) Fonti consultate
- Repo locale SOKE + README upstream
- ArXiv: https://arxiv.org/abs/2411.17799
- PDF scaricato localmente: `docs/SOKE_2411.17799.pdf`

## 9) Allineamento operativo (verifica workspace 2026-03-28)
### Confermato adesso
- Checkpoint presenti e leggibili in `experiments/mgpt/SOKE/checkpoints/` (last, last-v1, last-v2, min-*).
- Training precedente ha prodotto `experiments/mgpt/SOKE/csv_logs/metrics.csv` e checkpoint fino a epoca iniziale.
- Script inferenza completa presente: `scripts/run_inference_complete.sh`.
- Config inferenza completa presente: `configs/soke_infer_complete.yaml` (`TEST.SAVE_PREDICTIONS: True`).

### Stato inferenza completa
- Nei run verificati (`logs/infer_last_*.log`), la pipeline arriva a:
  - dataset load,
  - VAE load,
  - AMG checkpoint load,
  - avvio `Evaluating TM2TMetrics - Replication 0`.
- In questo ambiente di lavoro il test cade in fallback CPU (`CUDA not available. Falling back to CPU.`) e non risultano ancora artefatti finali `test_scores.json` / `*.pkl` sotto `results/mgpt/SOKE_INFER/...`.
- Quindi: **pipeline pronta, run completo su test set ancora da finalizzare su GPU reale**.

### Comando ufficiale da eseguire su macchina GPU
```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
bash scripts/run_inference_complete.sh last
```
Oppure confronto multi-checkpoint:
```bash
bash scripts/run_inference_complete.sh all
```

### Criterio di "done" inferenza completa
1. Presenza di `results/mgpt/SOKE_INFER/test_rank_0/test_scores.json`.
2. Presenza di file `results/mgpt/SOKE_INFER/test_rank_0/*.pkl`.
3. Log di inferenza che arriva a fine test senza interrupt.

## 10) Dataset privato HF + auto-sync (nuovo, 2026-03-29)
### Implementato
- Nuovo modulo: `mGPT/utils/dataset_autodownload.py`
  - controlla la presenza dei file minimi dataset (How2Sign/CSL/Phoenix + mean/std),
  - se mancanti, usa repo HF privata via variabile `SOKE_HF_DATASET_REPO`,
  - tenta `git clone`/`git pull` + `git lfs pull`,
  - ricontrolla i file richiesti e fallisce con errore esplicito se incompleto.
- Integrazione nel runtime:
  - `mGPT/data/H2S.py` ora invoca `ensure_dataset_available(cfg)` prima del `torch.load(mean/std)`.
  - Effetto: train e test usano auto-sync senza cambiare comandi principali.
- Script di upload dataset privato:
  - `scripts/hf_dataset_push_private.sh <USER/REPO> [LOCAL_DATA_DIR]`
  - inizializza git, imposta origin HF datasets, abilita git-lfs, traccia estensioni grandi, commit/push.
- Script bootstrap download (Docker-friendly):
  - `scripts/download_dataset_from_hf.sh <USER/REPO> [TARGET_DIR]`
  - usa `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` da env (nessuna chiave nel codice repo).
- Auto-sync ora preferisce `huggingface_hub.snapshot_download` (token env) e usa git-lfs solo come fallback.

### Variabili operative
- `SOKE_DATA_ROOT` (consigliato: `/home/cirillo/Desktop/SOKE_DATA`)
- `SOKE_H2S_ROOT`, `SOKE_CSL_ROOT`, `SOKE_PHOENIX_ROOT` (via helper script)
- `SOKE_HF_DATASET_REPO=<USER>/<DATASET_REPO_PRIVATA>`

### Documentazione aggiornata
- `DATASET_PRIVATE_HF.md` (flusso upload + auto-download)
- `README.md` (nota auto-download HF)

## 11) Dockerizzazione (nuovo, 2026-03-29)
### File aggiunti
- `Dockerfile`
- `docker-compose.yml`
- `docker/entrypoint.sh`
- `.env.example`
- `DOCKER.md`

### Comportamento
- Entry-point container:
  1. imposta path dataset (`/workspace/SOKE_DATA`),
  2. se `SOKE_HF_DATASET_REPO` e' valorizzata, lancia `scripts/download_dataset_from_hf.sh` (usa `HF_TOKEN` da env),
  3. avvia `train` o `infer`.
- Nessuna chiave hardcoded nella repo.

### Deploy remoto previsto
- Build locale/test locale con `docker compose`.
- Publish immagine su registry privato (consigliato GHCR).
- Pull su server remoto via SSH + `docker pull` e run con env vars/token.
