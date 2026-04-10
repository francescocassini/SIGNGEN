# Manifesto Di Riallineamento - SOKE

Data: 2026-04-10  
Workspace: `/home/cirillo/Desktop/SIGNGEN/SOKE`

## 1) Commit di riferimento
- Repo locale: `0841c0f5c8c3f37ca2d963823dd90d85fe095bd2`
- Repo ufficiale clonata: `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00` (`/tmp/SOKE_OFFICIAL`)

## 2) Ripristino codice/config a upstream
Eseguito restore byte-identico dei file training-critical da `/tmp/SOKE_OFFICIAL`.
Verifica effettuata con confronto `cmp` su file chiave: esito `ALL_RESTORED_IDENTICAL`.

Backup pre-ripristino:
- `_backup_pre_strict_restore/20260410_074400/`

## 3) Hash di parita' artefatti fondamentali
- mBART `pytorch_model.bin`
  - locale: `9894f84c379d7830f69fd1a95026869149cd5a2547a4a60b70cff4bd32b102fb`
  - ufficiale: `9894f84c379d7830f69fd1a95026869149cd5a2547a4a60b70cff4bd32b102fb`
- tokenizer DETO `tokenizer.ckpt`
  - locale: `1b225a317af368a84a37788be07c88af76582223c821d3c55734f73d6a18316c`
  - ufficiale: `1b225a317af368a84a37788be07c88af76582223c821d3c55734f73d6a18316c`
- `mean.pt`
  - locale attuale: `38041ea364a5b1e187f7d633edd3ae564e1f8771506b7390fee48e80e3de0c23`
  - ufficiale: `38041ea364a5b1e187f7d633edd3ae564e1f8771506b7390fee48e80e3de0c23`
- `std.pt`
  - locale attuale: `88ed5e7e9f484d35908088a22295fb6a8c05e1ad50f6dcf2018a253aa6bffee4`
  - ufficiale: `88ed5e7e9f484d35908088a22295fb6a8c05e1ad50f6dcf2018a253aa6bffee4`

## 4) Correzione split CSL (2 sample vuoti)
Rimossi dallo split train CSL i sample senza pose:
- `S003751_P0000_T00`
- `S005362_P0000_T00`

File modificato:
- `/home/cirillo/Desktop/SOKE_DATA/CSL-Daily/csl_clean.train`

Backup split prima della modifica:
- `/home/cirillo/Desktop/SOKE_DATA/CSL-Daily/csl_clean.train.pre_drop2_20260410_083133`

Conta attuale split CSL train:
- `18399` sample

## 5) Rigenerazione motion codes da zero
Pulizia completa:
- rimossa e ricreata `/home/cirillo/Desktop/SOKE_DATA/How2Sign/TOKENS_h2s_csl_phoenix`

Comando eseguito:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
CUDA_VISIBLE_DEVICES=0 python3 -u -m get_motion_code \
  --cfg configs/soke.yaml --nodebug --use_gpus 0 --device 0
```

Log run:
- `logs/get_motion_code_strict_20260410.log`

Esito run:
- completato (`Motion tokenization done...`)

Coverage token train post-rigenerazione:
- How2Sign: `30684 / 30965` (99.09%)
- CSL: `18399 / 18399` (100%)
- Phoenix: `7092 / 7092` (100%)
- Totale `.npy`: `56175`

## 6) Stato finale riallineamento
Riallineati i componenti critici del training:
- codice/config upstream,
- pesi mBART,
- checkpoint tokenizer DETO,
- normalizzazione ufficiale (`mean/std`),
- motion codes rigenerati da zero.

Nota tecnica residuale:
- How2Sign train resta a `99.09%` per campioni problematici lato pose (comportamento coerente con il dataset effettivo).
