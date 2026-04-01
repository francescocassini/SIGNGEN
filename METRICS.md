# METRICS - SOKE AMG (paper + mapping pratica)

Ultimo aggiornamento: 2026-04-01
Fonte principale: `docs/SOKE_ICCV2025_paper.pdf` (ICCV 2025, Signs as Tokens)

## 1) Scope (solo AMG)
Questo file raccoglie tutto cio che nel paper serve per valutare se il training della parte AMG (autoregressive multilingual generator) sta andando bene.

Nota: nel paper AMG non e trainato da zero assoluto, ma fine-tuned da `mBART-large-cc25`.

## 2) Setup training AMG riportato nel paper
Estratto da Sezione 4 (Implementation Details):

- Backbone AMG: `mBART-large-cc25`
- Struttura LM: encoder-decoder, 12 layer, hidden size 1024
- Loss training AMG: cross-entropy standard
  - Formula paper: `L_LM = -log P(Y|h_en)`
- Batch size AMG: `32 per GPU`
- Epoche AMG: `150`
- Optimizer AMG: `AdamW`
- LR scheduler AMG: `cosine`, start LR `2e-4`
- GPU usate nel paper: `6 x RTX 3090`
- Hyperparam di decoding multi-head: `lambda = 1/3`

Riferimenti nel testo estratto (`/tmp/soke_iccv2025.txt`): righe ~342-365.

## 3) Metriche usate nel paper per valutare AMG
## 3.1 Metriche principali (quantitative)
Per Text-to-Sign (SLG), il paper usa:

- `DTW-PA-JPE` (lower is better)
  - riportato separatamente per `Body` e `Hand`
- `DTW-JPE` (lower is better)
  - riportato separatamente per `Body` e `Hand`
- `B-T BLEU-4` (higher is better)
  - back-translation score, misura interpretabilita semantica

Riferimento: tabella 1 + sezione Evaluation Metrics (righe ~331-339, ~413-426).

## 3.2 Metriche secondarie (non training core, ma utili)
- `Latency (s/video)` nelle ablation di decoding (tabella 2)
- User study score (1-10) con signer professionisti (sezione 4.3)

## 4) Target numerici SOKE (AMG) dal paper
Questi sono i numeri target della riga `SOKE (ours)` in Tabella 1.

## 4.1 How2Sign (ASL)
- DTW-PA-JPE: Body `6.82`, Hand `2.35`
- DTW-JPE: Body `7.75`, Hand `10.08`
- B-T BLEU-4: `14.48`

## 4.2 CSL-Daily (CSL)
- DTW-PA-JPE: Body `6.24`, Hand `1.71`
- DTW-JPE: Body `7.38`, Hand `9.68`
- B-T BLEU-4: `11.30`

## 4.3 Phoenix-2014T (DGS)
- DTW-PA-JPE: Body `4.77`, Hand `1.38`
- DTW-JPE: Body `6.04`, Hand `7.72`
- B-T BLEU-4: `11.87`

Riferimento: tabella 1 (righe ~413-424).

## 5) Ablation AMG molto utili per capire se "sta andando bene"
## 5.1 Decoding + retrieval (Tabella 2)
Configurazione migliore del paper: `Multi-head + Retrieval`.

How2Sign:
- Avg `3.34`, Body `6.82`, Hand `2.35`, Latency `1.55 s/video`

CSL-Daily:
- Avg `2.72`, Body `6.24`, Hand `1.71`, Latency `1.52 s/video`

Phoenix-2014T:
- Avg `2.13`, Body `4.77`, Hand `1.38`, Latency `1.51 s/video`

Insight del paper:
- retrieval riduce average DTW di circa `19.9% / 19.6% / 23.9%` su H2S/CSL/Phoenix.

## 5.2 Scalabilita multi-lingua (Tabella 3)
Aggiungere dataset/language nel training migliora AMG:

- Solo H2S -> H2S DTW Body/Hand: `7.92 / 3.07`
- H2S + CSL -> H2S `7.11 / 2.63`, CSL `6.79 / 2.14`
- H2S + CSL + Phoenix -> H2S `6.82 / 2.35`, CSL `6.24 / 1.71`, Phoenix `4.77 / 1.38`

Interpretazione pratica:
- l'impostazione multilingual e parte del guadagno AMG, non solo tokenizer.

## 6) Cosa monitorare durante il nostro training AMG (operativo)
Per allinearci al paper e capire se stiamo convergendo bene, le metriche minime da tracciare ad ogni validazione sono:

- `how2sign_DTW_MPJPE_PA_lhand` (min)
- `csl_DTW_MPJPE_PA_lhand` (min)
- `phoenix_DTW_MPJPE_PA_lhand` (min)
- `how2sign_DTW_MPJPE_PA_body` (min)
- `csl_DTW_MPJPE_PA_body` (min)
- `phoenix_DTW_MPJPE_PA_body` (min)
- `BLEU_4` back-translation (max), se pipeline BT e attiva

Perche:
- il paper enfatizza DTW su body/hand e BLEU-4 come metrica semantica.
- il training locale salva gia checkpoint best su metriche DTW hand per dataset (vedi callback).

## 7) Mapping paper -> nomi metriche nel codice locale
Nel vostro codice (callback/metriche):

- Paper `DTW-PA-JPE Hand` ~
  - `Metrics/how2sign_DTW_MPJPE_PA_lhand`
  - `Metrics/csl_DTW_MPJPE_PA_lhand`
  - `Metrics/phoenix_DTW_MPJPE_PA_lhand`

- Paper `DTW-PA-JPE Body` ~
  - `Metrics/how2sign_DTW_MPJPE_PA_body`
  - `Metrics/csl_DTW_MPJPE_PA_body`
  - `Metrics/phoenix_DTW_MPJPE_PA_body`

- Paper `B-T BLEU-4` ~ `Metrics/Bleu_4`

Attenzione importante (codice attuale):
- In `mGPT/metrics/t2m.py`, e presente una nota che con `align_idx=0` la metrica e di fatto `DTW-JPE`, mentre il nome variabile resta `DTW_MPJPE_PA_*`.
- Quindi i nomi nei log/checkpoint possono non riflettere perfettamente la definizione matematica PA-vs-non-PA.

## 8) Cose NON specificate dal paper (gap da gestire noi)
Il paper non specifica in dettaglio:

- criterio preciso di early stopping/checkpoint selection
- frequenza validazione (every N epoch/steps)
- eventuale warmup LR
- gradient clipping
- weight decay esplicito
- seed/multi-run variance

Quindi, per confronto corretto paper-vs-nostro training, dobbiamo fissare internamente questi punti nel runbook.

## 9) Riferimenti file locali usati per questa estrazione
- `docs/SOKE_ICCV2025_paper.pdf`
- testo estratto: `/tmp/soke_iccv2025.txt`
- callback metriche/checkpoint: `mGPT/callback.py`
- implementazione metrica DTW: `mGPT/metrics/t2m.py`
