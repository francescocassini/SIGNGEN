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

## 2026-03-30
### Fatto
- Validato in modo end-to-end il flusso dataset privata HF in ambiente locale:
  - upload completato su repo `Francesco77/soke-private-data`,
  - download completato via `scripts/download_dataset_from_hf.sh`,
  - estrazione automatica archivi riuscita,
  - train avviabile dopo bootstrap dataset.
- Adottata strategia **archive-first** per evitare push ingestibile di milioni di file raw:
  - creati e pubblicati `How2Sign.tar.gz`, `CSL-Daily.tar.gz`, `Phoenix_2014T.tar.gz`.
- Aggiornati strumenti operativi:
  - nuovo `scripts/create_dataset_archives.sh`,
  - nuovo `scripts/hf_dataset_push_archives.sh`,
  - `scripts/hf_dataset_push_private.sh` con guard rail contro raw push massivo,
  - auto-extract archivi in:
    - `scripts/download_dataset_from_hf.sh`,
    - `mGPT/utils/dataset_autodownload.py`.

### Impatto operativo
- Obiettivo dataset HF sbloccato: dataset online, scaricabile e riproducibile senza setup manuale dei file raw.
- Ridotto drasticamente il rischio di upload estremamente lento dovuto a cardinalita' file elevata.

### Prossimi passi operativi
1. Validare identico flusso dentro Docker (`docker compose run --rm soke train` con `SOKE_HF_DATASET_REPO` e `HF_TOKEN`).
2. Implementare obiettivo 3: callback ogni 50 epoche con inferenza subset + GIF GT vs Pred.
3. Implementare obiettivo 4: logging GPU avanzato e report multi-GPU per rank.

### Nota sicurezza
- Se un token HF viene esposto in terminale/chat, ruotarlo subito da Hugging Face settings.

---

## Template aggiornamento rapido (da appendere ogni sessione)
- Data/ora:
- Cosa e' stato eseguito:
- Esito:
- Problemi trovati:
- Azioni correttive:
- Prossimo step:
## 2026-03-30 (sera) - Hotfix Docker train

- Pinned `transformers==4.41.2` in `requirements.txt` (compat con `torch 2.3.1` della Docker image).
- Aggiunto `sentencepiece>=0.1.99` in `requirements.txt` (richiesto da `MBartTokenizer`).
- Risolto problema split file mancanti in Docker causato da symlink assoluti:
  - `mGPT/utils/dataset_autodownload.py` ora ripara automaticamente i file split copiandoli da `data/splits/`.
  - `scripts/download_dataset_from_hf.sh` fa la stessa riparazione post download/extract.
- `scripts/create_dataset_archives.sh` usa `tar --dereference` per evitare di archiviare symlink (solo file reali).

## 2026-04-01
### Riconnessione repo + hardening workflow
- Repo locale ricollegata a `origin/main` (`https://github.com/francescocassini/SIGNGEN.git`) dopo incidente su `.git`.
- Verificato che artefatti pesanti (`deps`, `experiments`, `results`, `logs`, checkpoint) non sono in tracking Git.
- Aggiornata pipeline Docker per usare volume artefatti esterno (`SOKE_ARTIFACTS_ROOT_HOST`) in modo consistente locale/remoto.

### Fix bloccanti emersi nei test reali
- Errore Telegram `curl: command not found` nel container:
  - fix: aggiunto `curl` nel `Dockerfile`.
- Inferenza senza checkpoint:
  - fix in `test.py` con fallback checkpoint + supporto `SOKE_DEFAULT_TEST_CKPT`.
- Crash resume per assenza `wandb/latest-run`:
  - fix in `mGPT/config.py` (`resume_config`) per rendere W&B opzionale.

### Nuovo orchestratore fasi (train/validation/test sequenziali)
- Implementato `scripts/run_train_infer_cycles.sh`:
  - esegue train a target epoch,
  - esegue infer/test su `last.ckpt`,
  - riprende train dal checkpoint fino al prossimo target.
- Aggiunto mode `cycle` in `docker/entrypoint.sh`.
- GPU e device ora configurabili via `.env`:
  - `SOKE_USE_GPUS`, `SOKE_DEVICE_IDS`,
  - `SOKE_TRAIN_USE_GPUS`, `SOKE_TRAIN_DEVICE_IDS`,
  - `SOKE_TEST_USE_GPUS`, `SOKE_TEST_DEVICE_IDS`,
  - `SOKE_NUM_NODES`.
- Inferenzа script aggiornato per supportare device multipli:
  - `scripts/run_inference_complete.sh`.

### Nuove variabili `.env` introdotte
- Ciclo:
  - `SOKE_MODE=cycle`
  - `SOKE_TOTAL_EPOCHS`
  - `SOKE_CYCLE_EPOCHS`
  - `SOKE_CYCLE_RUN_INFER`
  - `SOKE_CYCLE_TEST_MAX_SAMPLES`
  - `SOKE_CYCLE_TEST_SKIP_METRICS`
- Runtime:
  - `SOKE_DEFAULT_TEST_CKPT`
  - `SOKE_TRAIN_RESUME` (gestito da orchestratore)
  - `SOKE_TRAIN_END_EPOCH`, `SOKE_VAL_EVERY_EPOCHS`, `SOKE_TEST_SKIP_METRICS`

### Note operative confermate
- Con `SOKE_TOTAL_EPOCHS=1` e checkpoint preesistente, il train puo' terminare subito per resume a target gia' raggiunto.
- Per run fresh: cancellare `.../SOKE_ARTIFACTS/experiments/mgpt/SOKE`.
- Le GIF vengono generate in `.../SOKE_ARTIFACTS/gifs` e inviate via Telegram (possibile delay di consegna).
- Se il dataset e' gia' presente, impostare `SOKE_AUTO_DOWNLOAD_DATASET=0` per evitare bootstrap ripetuti.

### Comando ufficiale fase attuale
```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
docker compose build --no-cache soke
docker compose run --rm soke cycle
```

## 2026-04-03
### Verifica stato reale ("dove siamo arrivati")
- Verificato che il repo `SOKE` e' pulito (`main...origin/main`) e che gli artefatti runtime non sono dentro al repo.
- Verificati artefatti su volume esterno:
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/experiments`
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/results`
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/gifs`
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/run_state`
- Run `cycle_20260401_184518` confermata con `status=success` in `manifest.txt`.
- Training GPU confermato fino a fine epoca 0 (1756 step) da log:
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/experiments/mgpt/SOKE/log_2026-04-01-18-45-22_train.log`
- Checkpoint presenti:
  - `last.ckpt` e `last-v1..last-v8` in `/home/cirillo/Desktop/SOKE_ARTIFACTS/experiments/mgpt/SOKE/checkpoints/`.
- Inferenza completata su subset:
  - `test_scores.json` + 1 file `.pkl` in `/home/cirillo/Desktop/SOKE_ARTIFACTS/results/mgpt/SOKE_INFER/test_rank_0/`.
- GIF GT vs Pred generate:
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/gifs/last_20260401_191206/`.

### Stato obiettivi (checkpoint rapido)
- Docker + dataset HF privata: operativo.
- Orchestrazione `cycle` train->infer: operativa (1 run validata).
- Inferenza full test set multi-dataset: ancora da finalizzare (nel run verificato CSL/Phoenix risultano 0).
- Automazione periodica GIF ogni 50 epoche: parziale, da consolidare su run lunghi.
