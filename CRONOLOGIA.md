# CRONOLOGIA - SOKE

## 2026-03-28
### Fatto
- Audit completo artefatti SOKE (checkpoint AMG, tokenizer, mBART, dataset readiness).
- Confermato che esistono checkpoint AMG utilizzabili subito:
  - `last.ckpt`, `last-v1.ckpt`, `last-v2.ckpt`, `min-*.ckpt`.
- Eseguito smoke test inferenza su `test.py`.
- Identificati e risolti bug bloccanti in `test.py`:
  - gestione `cfg.DEVICE` (ListConfig),
  - blocco post-main fuori scope,
  - fallback CPU se GPU non disponibile.
- Scaricato paper SOKE in locale:
  - `docs/SOKE_2411.17799.pdf`.
- Creato `HANDOFF.md` con stato/piano operativo end-to-end.
- Aggiunta config dedicata inferenza completa:
  - `configs/soke_infer_complete.yaml` (`SAVE_PREDICTIONS=True`).
- Aggiunto supporto CLI checkpoint:
  - `--checkpoint` in `mGPT/config.py`.
- Creato script operativo:
  - `scripts/run_inference_complete.sh` (`last` | `all` | path checkpoint).
- Validato workflow con run controllato:
  - il test arriva a `Evaluating TM2TMetrics` e procede oltre il loading checkpoint.

### Evidenza tecnica
- Inferenza arriva a:
  - model load,
  - VAE tokenizer load,
  - checkpoint AMG load,
  - avvio `trainer.test(...)`.
- Dataset test caricati con count non-zero:
  - How2Sign 2308, CSL 1176, Phoenix 642.

### Stato decisionale
- Decisione: si puo' procedere con inferenza subito, anche se non ancora ottimizzata.
- Decisione: serve una fase di hardening metriche/val prima del benchmark finale.

### Prossimi passi operativi
1. Eseguire test completo su `last.ckpt` (senza timeout) e salvare metriche finali.
2. Eseguire test confronto su almeno un `min-*.ckpt`.
3. Attivare `TEST.SAVE_PREDICTIONS=True` e generare sample per render.
4. Render (`vis_mesh`/Blender) su subset fisso per valutazione qualitativa.
5. Consolidare config inferenza dedicata (`configs/soke_infer.yaml`).
6. Lanciare `scripts/run_inference_complete.sh all` su GPU reale e raccogliere report finale.

### Aggiornamento allineamento conversazione/workspace (stessa data)
- Verificato stato reale file chiave: `HANDOFF.md`, `CRONOLOGIA.md`, `configs/soke_infer_complete.yaml`, `scripts/run_inference_complete.sh`, `test.py`, `mGPT/config.py`.
- Verificato che i due run inferenza presenti nei log (`infer_last_20260328_062722.log`, `infer_last_20260328_064810.log`) arrivano a `Evaluating TM2TMetrics - Replication 0`.
- Verificato che in ambiente corrente manca GPU disponibile per Lightning (`CUDA not available. Falling back to CPU.`), quindi non sono stati ancora prodotti output finali completi (`test_scores.json` / `*.pkl`) in `results/mgpt/SOKE_INFER/test_rank_0`.
- Confermato che checkpoint AMG sono disponibili e pronti per run completo su GPU:
  - `last.ckpt`, `last-v1.ckpt`, `last-v2.ckpt`, `min-*.ckpt`.

### Next immediate action (bloccante)
1. Eseguire `bash scripts/run_inference_complete.sh last` su sessione con GPU visibile.
2. Verificare output con:
   - `ls -lah results/mgpt/SOKE_INFER/test_rank_0 | head`
   - `ls results/mgpt/SOKE_INFER/test_rank_0/*.pkl | head`
3. Solo dopo questo step: analisi qualitativa e confronto `last` vs `min-*`.

### Miglioria observability inferenza (nuovo)
- Aggiornato `test.py` con callback `InferenceProgressLogger` per avere log testuali durante `trainer.test(...)` anche quando la progress bar Rich non e' leggibile nel file log.
- Nuovi indicatori in log:
  - batch completati / batch totali,
  - percentuale avanzamento,
  - velocita' batch/s,
  - elapsed time,
  - ETA,
  - riga finale di completamento con tempo totale.

### Miglioria preview rapida (nuovo)
- Aggiunte opzioni CLI test:
  - `--test_max_samples` per limitare i sample del test set,
  - `--skip_metrics` per saltare metriche costose (DTW/SMPL-X) e vedere subito output.
- `mGPT/data/__init__.py`: `test_dataloader()` ora applica `TEST.MAX_SAMPLES` via `Subset`.
- `mGPT/data/humanml/dataset_t2m.py`: in preview (`max_samples`) interrompe in anticipo lo scan dei dataset quando raggiunge N sample, riducendo startup time.
- `mGPT/models/mgpt.py`: path test veloce senza `feats2joints`/metriche quando `SKIP_METRICS=True`.
- `mGPT/models/base.py`: `on_test_epoch_end()` compatibile con `SKIP_METRICS`, scrive `test_scores.json` informativo e usa rank-safe anche senza DDP.
- `scripts/run_inference_complete.sh`: supporta env vars:
  - `MAX_SAMPLES=<N>`
  - `SKIP_METRICS=1`
- Test smoke verificato:
  - comando con `--test_max_samples 2 --skip_metrics` termina correttamente e produce log di avanzamento + completamento.

## 2026-03-29
### Fatto
- Implementato auto-download dataset da Hugging Face (repo privata) quando file dataset sono mancanti:
  - nuovo file `mGPT/utils/dataset_autodownload.py`.
- Integrato auto-check/sync nel datamodule:
  - `mGPT/data/H2S.py` chiama `ensure_dataset_available(cfg)` prima di caricare mean/std.
- Aggiunto script upload dataset privato:
  - `scripts/hf_dataset_push_private.sh`.
- Aggiornata documentazione:
  - `DATASET_PRIVATE_HF.md` (upload + auto-download),
  - `README.md` (variabile `SOKE_HF_DATASET_REPO`).
- Aggiunto script esplicito di bootstrap dataset (utile per Docker entrypoint):
  - `scripts/download_dataset_from_hf.sh`.
- Auto-download esteso con preferenza `huggingface_hub.snapshot_download` (supporta `HF_TOKEN`) con fallback git-lfs.

### Impatto operativo
- Se la directory dataset locale e' vuota/incompleta, train/test tentano sync da HF automaticamente.
- Il dataset resta esterno alla repo codice SOKE.

### Dockerizzazione (nuovo)
- Aggiunti file:
  - `Dockerfile`
  - `docker-compose.yml`
  - `docker/entrypoint.sh`
  - `.env.example`
  - `DOCKER.md`
- Design operativo:
  - il container usa `SOKE_DATA_ROOT=/workspace/SOKE_DATA` montato da host,
  - in entrypoint esegue bootstrap dataset da HF privata (`scripts/download_dataset_from_hf.sh`) se `SOKE_HF_DATASET_REPO` e' presente,
  - poi avvia `train` o `infer` con comandi standard.
- Sicurezza credenziali:
  - token HF via env (`HF_TOKEN`), non nel codice;
  - `.env` ignorato da git (`.gitignore`).

---

## Template aggiornamento rapido (da appendere ogni sessione)
- Data/ora:
- Cosa e' stato eseguito:
- Esito:
- Problemi trovati:
- Azioni correttive:
- Prossimo step:
