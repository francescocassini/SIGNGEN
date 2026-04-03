# HANDOFF - SOKE (Signs as Tokens)

Ultimo aggiornamento: 2026-04-03 (Europe/Rome)

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

## 12) Piano aggiornato (richiesta utente, 2026-03-29)
### Obiettivo 1: Docker scarica dataset HF privata (38GB)
- Stato: **COMPLETATO (locale + runtime), da validare in Docker end-to-end**
- Fatto:
  - auto-sync runtime in `mGPT/utils/dataset_autodownload.py`,
  - script bootstrap `scripts/download_dataset_from_hf.sh`,
  - entrypoint Docker che avvia bootstrap dataset se `SOKE_HF_DATASET_REPO` e' impostata,
  - upload HF completato con strategia archive-first (`How2Sign.tar.gz`, `CSL-Daily.tar.gz`, `Phoenix_2014T.tar.gz`),
  - download reale verificato: snapshot da HF + estrazione locale completata con successo.
- Manca:
  - validazione esplicita dello stesso flusso dentro container Docker.

### Obiettivo 2: Docker esegue SOKE come in locale
- Stato: **PARZIALE AVANZATO**
- Fatto:
  - `Dockerfile`, `docker-compose.yml`, `docker/entrypoint.sh`,
  - comandi `train` e `infer` in entrypoint.
- Manca:
  - test end-to-end reale su macchina utente (build + run train + run infer),
  - eventuale tuning multi-GPU (default attuale single GPU).

### Obiettivo 3: Ogni 50 epoche, inferenza subset test + GIF GT vs Pred
- Stato: **MANCANTE**
- Fatto:
  - esiste tool rendering `scripts/preview_test_sample.py`,
  - esiste pipeline inferenza `scripts/run_inference_complete.sh`.
- Manca:
  - callback train `every_n_epochs=50` che lanci test su subset,
  - post-process automatico da `.pkl` a GIF confronto GT vs Pred,
  - directory output standard per artefatti periodici.

### Obiettivo 4: Logging training avanzato (stato, loss/metriche, GPU, multi-GPU)
- Stato: **PARZIALE**
- Fatto:
  - progress bar + callback metriche in `mGPT/callback.py`,
  - CSV logger in `train.py`,
  - supporto DDP se `DEVICE` contiene piu' GPU.
- Manca:
  - logger GPU integrato nel training log (util/mem/temp/power),
  - report chiaro per rank/GPU in multi-GPU,
  - dashboard operativa unica (W&B o log parser) pronta per produzione Docker.

## 13) Aggiornamento operativo 2026-03-30 (milestone HF dataset)
### Cosa e' stato chiuso
- Repo dataset privata HF ripulita e ripopolata in modalita' archive-first.
- Upload completato con successo (3 oggetti LFS grandi, no milioni di file piccoli):
  - `How2Sign.tar.gz`
  - `CSL-Daily.tar.gz`
  - `Phoenix_2014T.tar.gz`
- Verifica remota positiva: `list_repo_files` restituisce i 3 archivi + `.gitattributes`.
- Download locale verificato con:
  - `scripts/download_dataset_from_hf.sh`,
  - env `SOKE_HF_DATASET_REPO` + `HF_TOKEN`,
  - estrazione automatica archivi completata.
- Esito: dataset online, scaricabile, e riutilizzabile in train/test senza setup manuale dei file raw.

### Migliorie tecniche introdotte
- `scripts/create_dataset_archives.sh`: genera archivi dataset da root locale.
- `scripts/hf_dataset_push_archives.sh`: push HF solo archivi (flusso consigliato).
- `scripts/hf_dataset_push_private.sh`: guard rail contro push raw con milioni di file.
- `scripts/download_dataset_from_hf.sh`: estrazione automatica archivi dopo snapshot.
- `mGPT/utils/dataset_autodownload.py`: auto-extract archivi se presenti in data root.

### Priorita' successive (piano attivo)
1. **Obiettivo 2 (Docker e2e):** eseguire `docker compose run --rm soke train` su ambiente GPU e verificare bootstrap dataset da HF + start training.
2. **Obiettivo 3 (every 50 epochs):** implementare callback periodica inferenza subset test e pipeline GIF GT vs Pred.
3. **Obiettivo 4 (logging avanzato):** integrare telemetria GPU (util/mem/temp/power) nel log di training e report rank-aware in multi-GPU.
4. Usare `DOCKER_PARITY_MAP.md` come runbook unico per equivalenza locale vs Docker (train, monitor, inferenza, GIF).

## 14) Hotfix Docker training (2026-03-30 sera)
### Problemi osservati
1. Crash import in container:
   - `ImportError: cannot import name 'GenerationMixin' from transformers.generation`
   - causa: `transformers` non pinnato -> versione troppo nuova rispetto a `torch 2.3.1` dell'immagine base.
2. Falso "missing dataset files" dopo download/extract:
   - diversi split (`csl_clean.*`, `phoenix14t.*`, `how2sign_realigned_*.csv`) risultavano symlink assoluti verso path host (`/home/cirillo/...`),
   - nel container quei symlink erano rotti, quindi i file risultavano mancanti.

### Fix applicati
- `requirements.txt`
  - `transformers==4.41.2` (pin compatibile con stack torch Docker attuale).
  - `sentencepiece>=0.1.99` (necessario per `MBartTokenizer`).
- `mGPT/utils/dataset_autodownload.py`
  - aggiunta riparazione automatica split files da `data/splits/*` del repo quando mancanti/roken symlink.
  - riparazione eseguita sia prima del sync HF che dopo l'estrazione archivi.
- `scripts/download_dataset_from_hf.sh`
  - aggiunta stessa logica di riparazione split post-download/post-extract.
- `scripts/create_dataset_archives.sh`
  - `tar --dereference` per includere file reali negli archivi futuri (non symlink).

### Impatto
- train in Docker non dovrebbe piu' fermarsi su `GenerationMixin`,
- dataset check non dovrebbe piu' fallire per symlink spezzati,
- i prossimi archivi HF saranno robusti anche su host/container diversi.
- container ora eseguito con UID/GID host (`docker-compose.yml`) per evitare file root bloccati in `SOKE_DATA`.
- nuovo helper `scripts/init_docker_env.sh` per generare `.env` con `LOCAL_UID/LOCAL_GID` automatici (setup replicabile per altri utenti).

### Nota sicurezza operativa
- Evitare di incollare token HF in log/chat/versioning.
- Se un token e' stato esposto, rigenerarlo immediatamente dal pannello Hugging Face.

## 15) Stato operativo finale (2026-03-31)
### Esito attuale
- Docker training avviabile end-to-end con GPU.
- Download dataset da HF privata funzionante (snapshot + extract).
- Dopo reset completo `SOKE_DATA`, il bootstrap ricrea i dati correttamente.
- Fix aggiuntivo 2026-03-31: estrazione archivi non viene piu' saltata se mancano `CSL-Daily/mean.pt` e `CSL-Daily/std.pt`.

### Hardening permessi (anti-file bloccati)
- `docker-compose.yml` esegue il container con user host:
  - `user: "${LOCAL_UID:-1000}:${LOCAL_GID:-1000}"`
- Cache HF container su mount utente:
  - `HF_HOME=/workspace/.cache/huggingface`
  - bind su `${HOME}/.cache/huggingface:/workspace/.cache/huggingface`
- `docker/entrypoint.sh` imposta:
  - `umask ${SOKE_UMASK:-0002}`
  - log esplicito uid/gid/umask all'avvio.

Impatto: i file scritti in `SOKE_DATA` devono risultare cancellabili dall'utente host senza root.

### Hardening runtime cache/temp
- In `docker-compose.yml` aggiunte env:
  - `HOME=/workspace/SOKE`
  - `TMPDIR=/tmp`
  - `XDG_CONFIG_HOME=/tmp/.config`
  - `MPLCONFIGDIR=/tmp/matplotlib`
- In `docker/entrypoint.sh` creazione preventiva directory cache/temp/config.

Impatto: ridotti warning su matplotlib (`/.config`) e su file temporanei HF (`/workspace/tmp_*`).

### Setup `.env` standard (replicabile per altri utenti)
- Script ufficiale:
  - `scripts/init_docker_env.sh`
- Genera `.env` con:
  - `SOKE_HF_DATASET_REPO`
  - `HF_TOKEN` placeholder (da compilare manualmente)
  - `SOKE_DATA_ROOT_HOST`
  - `SOKE_AUTO_DOWNLOAD_DATASET`
  - `SOKE_TRAIN_CFG` / `SOKE_TEST_CFG`
  - `LOCAL_UID` / `LOCAL_GID` auto dalla macchina corrente.

Nota operativa:
- Anche chi fa pull su un altro PC deve creare il proprio `.env` locale (non versionato), preferibilmente via `scripts/init_docker_env.sh`.

### Procedura ufficiale: reset dataset + train
```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE

# stop/rm container SOKE
docker ps -q --filter "name=soke" | xargs -r docker stop
docker ps -aq --filter "name=soke" | xargs -r docker rm -f

# hard reset dataset
sudo chown -R "$USER:$USER" /home/cirillo/Desktop/SOKE_DATA || true
sudo chmod -R u+rwX /home/cirillo/Desktop/SOKE_DATA || true
sudo rm -rf /home/cirillo/Desktop/SOKE_DATA
mkdir -p /home/cirillo/Desktop/SOKE_DATA
chmod 775 /home/cirillo/Desktop/SOKE_DATA

# env + build + train
scripts/init_docker_env.sh
# edit .env and set HF_TOKEN
docker compose build --no-cache soke
docker compose run --rm soke train
```

### Monitor training
```bash
CID=$(docker ps --filter "name=soke-soke-run" --format "{{.ID}}" | head -n1)
docker logs -f "$CID"

watch -n 5 'ls -lah /home/cirillo/Desktop/SIGNGEN/SOKE/experiments/mgpt/SOKE/checkpoints'
tail -f /home/cirillo/Desktop/SIGNGEN/SOKE/experiments/mgpt/SOKE/csv_logs/metrics.csv
watch -n 1 "nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used,temperature.gpu --format=csv,noheader"
```

### Nota shell locale (Conda)
- In alcuni terminali con env Conda custom, comandi mancanti possono scatenare errore `command-not-found` (librerie `apt_pkg`/`GLIBCXX`).
- Evitare dipendenze non necessarie nei comandi operativi (es. usare filtri Docker nativi invece di pipeline con tool non presenti).

## 16) Aggiornamento operativo (2026-03-31, follow-up finale Docker)
### Esito sessione
- Verifica utente positiva: il flusso Docker (download dataset + bootstrap + run) risulta funzionante in pratica.
- Osservato comportamento cache HF coerente con `snapshot_download` (riuso cache locale quando presente).

### Allineamento log bootstrap Docker
- `docker/entrypoint.sh` aggiornato per evitare log fuorvianti:
  - `SOKE_AUTO_DOWNLOAD_DATASET` ora accetta anche `true/yes/on` (oltre a `1`).
  - messaggio di skip esplicita il motivo:
    - variabile auto-download disabilitata, oppure
    - `SOKE_HF_DATASET_REPO` non impostata.
- Impatto: diagnosi piu' chiara in fase di reset/riavvio da zero.

## 17) Prossima fase (obiettivi generici)
### O1 - Logging training strutturato
- Definire uno standard unico per log runtime/training (console + file + CSV/JSON).
- Tracciare stato run, loss/metriche principali, tempi epoca/step, resume info.
- Garantire leggibilita' sia locale sia Docker/multi-sessione.

### O2 - Gestione pesi/checkpoint
- Definire policy checkpoint (best/last, retention, naming, cleanup).
- Salvare metadati minimi per ripristino rapido (config, commit, seed, epoca).
- Preparare esportazione/artifact packaging dei pesi per deploy o condivisione.

### O3 - Notifiche Telegram durante training
- Inviare aggiornamenti periodici stato training (start, heartbeat, end/fail).
- Inviare eventi chiave (nuovo best checkpoint, errore run, stop inatteso).
- Allegare nel messaggio un riepilogo sintetico (loss/metriche/GPU/ETA) e link/path artefatti.

### Criterio di completamento fase
1. Pipeline logging + checkpoint + notifiche attivabile da config/env senza edit manuali codice.
2. Testata almeno una run completa con notifiche Telegram ricevute correttamente.
3. Documentazione operativa corta (setup token/chat_id + esempi comandi + troubleshooting).

## 18) Aggiornamento operativo (2026-04-01) - Pipeline locale/remote stabilizzata
### Esito sessione
- Flusso end-to-end validato in locale con GPU:
  - training avviato correttamente,
  - test/inferenza avviata correttamente,
  - notifiche Telegram funzionanti (testo + GIF, con possibile ritardo lato rete Telegram),
  - artefatti scritti su volume host esterno.

### Modifiche tecniche integrate
- `Dockerfile`:
  - aggiunto `curl` (necessario per `scripts/telegram_notify.sh` nel container).
- `test.py`:
  - fallback robusto checkpoint in inferenza:
    - supporto `SOKE_DEFAULT_TEST_CKPT`,
    - fallback automatico a `.../experiments/mgpt/SOKE/checkpoints/last.ckpt`,
    - errore esplicito se nessun checkpoint e' disponibile.
- `mGPT/config.py`:
  - resume robusto anche senza cartella `wandb/latest-run`,
  - override runtime da env (es. `SOKE_TRAIN_END_EPOCH`, `SOKE_TEST_MAX_SAMPLES`, `SOKE_TEST_SKIP_METRICS`, `SOKE_TRAIN_RESUME`).
- `docker/entrypoint.sh`:
  - mode `cycle` aggiunto,
  - train/test GPU selection env-driven (`SOKE_*_USE_GPUS`, `SOKE_*_DEVICE_IDS`, `SOKE_NUM_NODES`),
  - mantenuti marker run (`STARTED/DONE/FAILED`) e manifest.
- Nuovo orchestratore:
  - `scripts/run_train_infer_cycles.sh` implementa ciclo sequenziale:
    - train fino a target epoch,
    - inferenza su checkpoint `last.ckpt`,
    - resume training fino al prossimo target.
- `scripts/run_inference_complete.sh`:
  - supporto device multipli via env (`SOKE_TEST_USE_GPUS`, `SOKE_TEST_DEVICE_IDS`).
- Setup env/documentazione:
  - aggiornati `.env.example`, `scripts/init_docker_env.sh`, `DOCKER.md` con variabili ciclo/GPU.

### Decisione architetturale
- Disattivata come default la vecchia inferenza periodica in callback durante training sulla stessa GPU (`SOKE_PERIODIC_INFER_EVERY_N_EPOCHS=0`), per evitare OOM.
- Adottato ciclo sequenziale train -> infer -> resume (`mode=cycle`) per separare le fasi e liberare VRAM tra un processo e l'altro.

### Variabili `.env` operative (nuove/chiave)
- Selezione GPU/nodi:
  - `SOKE_USE_GPUS`, `SOKE_DEVICE_IDS`,
  - `SOKE_TRAIN_USE_GPUS`, `SOKE_TRAIN_DEVICE_IDS`,
  - `SOKE_TEST_USE_GPUS`, `SOKE_TEST_DEVICE_IDS`,
  - `SOKE_NUM_NODES`.
- Ciclo train/infer:
  - `SOKE_MODE=cycle`,
  - `SOKE_TOTAL_EPOCHS`,
  - `SOKE_CYCLE_EPOCHS`,
  - `SOKE_CYCLE_RUN_INFER`,
  - `SOKE_CYCLE_TEST_MAX_SAMPLES`,
  - `SOKE_CYCLE_TEST_SKIP_METRICS`.
- Checkpoint default inferenza:
  - `SOKE_DEFAULT_TEST_CKPT`.

### Runbook minimo validato
```bash
cd /home/cirillo/Desktop/SIGNGEN/SOKE
docker compose build --no-cache soke
docker compose run --rm soke cycle
```

### Note operative emerse
- Se `last.ckpt` esiste gia' e `SOKE_TOTAL_EPOCHS` non supera l'epoca gia' raggiunta, il train puo' chiudersi subito (resume a target gia' raggiunto).
- Per test "fresh" da epoca 0:
  - rimuovere `.../SOKE_ARTIFACTS/experiments/mgpt/SOKE` prima del run.
- Se il dataset e' gia' presente e stabile, impostare `SOKE_AUTO_DOWNLOAD_DATASET=0` per evitare bootstrap/estrazione ad ogni avvio.

## 19) Aggiornamento operativo (2026-04-03) - Stato reale raggiunto (verifica artefatti)
### Verifica oggettiva eseguita
- Workspace Git `SOKE` pulito (`main...origin/main`), senza modifiche locali non committate.
- Nel repo codice locale non ci sono artefatti runtime (`logs/`, `results/`, `experiments/.../checkpoints`), perche' gli output sono su volume esterno.
- Artefatti trovati e verificati in:
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/experiments`
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/results`
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/gifs`
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/run_state`

### Stato run confermato
- Run `cycle` conclusa con successo:
  - `run_state/cycle_20260401_184518/manifest.txt` riporta `status=success`,
  - commit registrato nel manifest: `a94de18`.
- Training su GPU realmente avviato e completato per 1 epoca:
  - log: `/home/cirillo/Desktop/SOKE_ARTIFACTS/experiments/mgpt/SOKE/log_2026-04-01-18-45-22_train.log`,
  - progresso fino a `step=1756/1756`, chiusura con `Training done`.
- Checkpoint presenti su volume esterno:
  - `last.ckpt` + `last-v1.ckpt` ... `last-v8.ckpt` in `/home/cirillo/Desktop/SOKE_ARTIFACTS/experiments/mgpt/SOKE/checkpoints/`.

### Inferenza/GIF confermate
- Inferenza completata su subset test (1 sample effettivo):
  - output `test_scores.json` presente:
    `/home/cirillo/Desktop/SOKE_ARTIFACTS/results/mgpt/SOKE_INFER/test_rank_0/test_scores.json`
  - output `.pkl` presente:
    `/home/cirillo/Desktop/SOKE_ARTIFACTS/results/mgpt/SOKE_INFER/test_rank_0/-fZc293MpJk_2-1-rgb_front.pkl`
- GIF GT vs Pred generate:
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/gifs/last_20260401_191206/-fZc293MpJk_2-1-rgb_front_pred.gif`
  - `/home/cirillo/Desktop/SOKE_ARTIFACTS/gifs/last_20260401_191206/-fZc293MpJk_2-1-rgb_front_compare_ref_pred.gif`

### Lettura sintetica "dove siamo arrivati"
- Obiettivo Docker+dataset HF: **CHIUSO (operativo)**.
- Obiettivo run cycle train->infer->resume: **CHIUSO (almeno 1 ciclo verificato)**.
- Obiettivo inferenza completa su test set intero multi-dataset: **NON ANCORA CHIUSO** (run verificata su subset rapido; CSL/Phoenix restano a 0 nel run osservato).
- Obiettivo GIF periodiche ogni 50 epoche: **PARZIALE** (GIF presenti, ma periodicita' automatica multi-epoca da consolidare su run lunghi).

### Prossimo step immediato consigliato
1. Lanciare un ciclo non-preview (senza limitazione sample) per ottenere metriche finali su test set pieno.
2. Confrontare `last.ckpt` vs almeno un `min-*.ckpt` con stessa procedura.
3. Aggiornare tabella benchmark finale (how2sign/csl/phoenix) solo dopo run full senza skip metriche.
