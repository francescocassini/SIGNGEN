# TRAINING Audit - Confronto Setup Locale vs SOKE Ufficiale

Data audit: 2026-04-10  
Workspace: `/home/cirillo/Desktop/SIGNGEN/SOKE`

## Commit di riferimento

| Voce | Valore |
|---|---|
| Repo locale (HEAD) | `0841c0f5c8c3f37ca2d963823dd90d85fe095bd2` |
| Repo ufficiale clonata (HEAD) | `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00` |
| Repo ufficiale clonata in | `/tmp/SOKE_OFFICIAL` |

## Tabella principale di confronto

| # | Punto di controllo | Setup ufficiale atteso | Stato locale | Esito | Evidenza concreta |
|---|---|---|---|---|---|
| 1 | Base model AMG | mBART-large-cc25 (release SOKE in `deps/mbart-h2s-csl-phoenix`) | Stesso pacchetto locale presente | `OK (Identico)` | SHA256 `pytorch_model.bin` ufficiale = locale: `9894f84c379d7830f69fd1a95026869149cd5a2547a4a60b70cff4bd32b102fb` |
| 2 | Tokenizer DETO checkpoint | Checkpoint tokenizer ufficiale usato in `TRAIN.PRETRAINED_VAE` | Checkpoint locale coincidente byte-per-byte | `OK (Identico)` | SHA256 ufficiale = locale: `1b225a317af368a84a37788be07c88af76582223c821d3c55734f73d6a18316c` |
| 3 | Motion codes generati con `get_motion_code` | `python -m get_motion_code --cfg configs/soke.yaml` con tokenizer ufficiale | Codici presenti ma coverage train non 100% su tutti i dataset | `PARZIALE` | H2S train `30684/30965` (99.09%), CSL train `18399/18401` (99.99%), Phoenix train `7092/7092` (100%) |
| 4 | Dataset e split ufficiali | How2Sign + CSL-Daily + Phoenix-2014T con split ufficiali | Split locali uguali agli split repo | `OK (Identico)` | Hash uguali per tutti i file split (How2Sign CSV, CSL gz, Phoenix gz) |
| 5 | Pose SMPL-X ufficiali + preprocessing | Pose rilasciate + mean/std ufficiali | Mean/std locali diversi da quelli ufficiali scaricati | `NO (Differente)` | `mean.pt` e `std.pt` con SHA256 diversi; `torch.equal=False`, `allclose=False` |
| 6 | Config `configs/soke.yaml` allineata | Config ufficiale SOKE | Config locale modificata | `NO (Differente)` | Diff su `DEVICE`, `PRETRAINED_VAE`, dataset root via env, `VAL_EVERY_STEPS`, ecc. |
| 7 | Retrieval-enhanced SLG | Variante finale con retrieval attivo | Retrieval attivo in `mgpt_mbart.py` (`num_kws_per_sen=3`) | `OK (Allineato)` | Uso di `scripts/name2kws_*.json` e `scripts/word2code.json` |
| 8 | Training stage AMG | `TRAIN.STAGE: lm_pretrain` | Presente nel config locale | `OK (Allineato)` | `configs/soke.yaml` locale |
| 9 | Procedura di test e metriche paper | DTW-PA-JPE, DTW-JPE, BT BLEU-4 su split test ufficiale | Artefatti metriche finali non disponibili in questo workspace | `NON DIMOSTRABILE` | `experiments/` locale senza output utili nel momento dell’audit |

## Similarita' confermate (forti)

| Categoria | Risultato | Dettaglio |
|---|---|---|
| Pesi mBART | Identici | File ufficiali scaricati e hash combacianti |
| Checkpoint tokenizer DETO | Identico | Hash combaciante tra ufficiale e locale |
| Split dataset | Identici | Hash combacianti su How2Sign/CSL/Phoenix |
| Retrieval assets | Identici | `word2code.json`, `name2kws_train/val/test.json` allineati |
| Stage AMG | Allineato | `lm_pretrain` confermato |

## Differenze trovate (impattano la riproducibilita')

| Area | Differenza osservata | Impatto |
|---|---|---|
| Preprocessing statistico | `mean.pt`/`std.pt` locali diversi dagli ufficiali | Alto: cambia la normalizzazione pose |
| Config training | `configs/soke.yaml` locale non identico all'upstream | Medio/Alto: cambia comportamento run |
| Implementazione locale | 16 file divergenti rispetto upstream | Medio/Alto: possibile variazione training/inferenza |
| Motion tokens train | Copertura non perfetta su H2S/CSL | Medio: training non su identico set tokenizzato |

## File principali divergenti tra repo locale e ufficiale

- `configs/assets.yaml`
- `configs/deto.yaml`
- `configs/soke.yaml`
- `mGPT/archs/lm_multihead.py`
- `mGPT/callback.py`
- `mGPT/config.py`
- `mGPT/data/H2S.py`
- `mGPT/data/__init__.py`
- `mGPT/data/humanml/dataset_t2m.py`
- `mGPT/data/humanml/dataset_t2m_cb.py`
- `mGPT/data/humanml/dataset_t2m_token.py`
- `mGPT/models/base.py`
- `mGPT/models/mgpt.py`
- `mGPT/utils/human_models.py`
- `mGPT/utils/load_checkpoint.py`
- `scripts/get_motion_code.py`

## Evidenze hash (estratto)

| File | SHA256 |
|---|---|
| `deps/mbart-h2s-csl-phoenix/pytorch_model.bin` | `9894f84c379d7830f69fd1a95026869149cd5a2547a4a60b70cff4bd32b102fb` |
| `deps/tokenizer_ckpt/tokenizer.ckpt` | `1b225a317af368a84a37788be07c88af76582223c821d3c55734f73d6a18316c` |
| `mean_official.pt` | `38041ea364a5b1e187f7d633edd3ae564e1f8771506b7390fee48e80e3de0c23` |
| `CSL-Daily/mean.pt` locale | `78b9ef34c4b4073d209cb597bd865ac6d59ebb90fa27ffddceb52b8a11ecf73a` |
| `std_official.pt` | `88ed5e7e9f484d35908088a22295fb6a8c05e1ad50f6dcf2018a253aa6bffee4` |
| `CSL-Daily/std.pt` locale | `8820481d607fbce8068d052634300e0e2fcbc7cdf1fbbd59fe74595300196914` |

## Verdetto sintetico

Il training locale e' **molto vicino** all'ufficiale su componenti fondamentali (mBART, tokenizer, split, retrieval), ma **non puo' essere dichiarato identico** per via di:
- mean/std diversi,
- config `soke.yaml` modificata,
- divergenze codice upstream,
- motion token coverage train non perfetta su tutti i dataset.


## Aggiornamento riallineamento (2026-04-10, chiusura)

Dopo l'audit iniziale, e' stato eseguito un riallineamento strict verso upstream.

### Cosa e' stato riallineato
- codice/config training-critical ripristinati byte-identici all'upstream;
- `mean.pt` e `std.pt` sostituiti con file ufficiali (hash identici);
- motion codes rigenerati da zero dopo clean completo della cartella token;
- split CSL train corretto rimuovendo 2 sample vuoti (con backup del file originale).

### Stato attuale
- mBART: allineato (hash match)
- tokenizer DETO: allineato (hash match)
- mean/std: allineati (hash match)
- motion codes train:
  - How2Sign `30684/30965`
  - CSL `18399/18399`
  - Phoenix `7092/7092`
  - totale `56175` file `.npy`

Manifesto completo:
- `RIALLINEAMENTO_MANIFEST.md`
