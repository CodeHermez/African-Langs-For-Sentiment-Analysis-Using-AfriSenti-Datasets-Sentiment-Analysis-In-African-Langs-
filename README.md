# Group 60 — AfriSenti Sentiment Analysis

**COS760 Natural Language Processing | University of Pretoria | 2026**
**Authors:** Praises Obi (u26819661) | Ishe Allen Chihobo (u22592238)

## Project Overview
Sentiment analysis across five African languages (Hausa, Yorùbá, Igbo,
Nigerian Pidgin, Swahili) using AfriSenti. Compares TF-IDF, mBERT, LAFT,
and MAFT with a novel code-mixing diagnostic split.

## Folder Structure
- `notebooks/` — 5 Kaggle notebooks, run in order NB1 to NB5
- `results/`   — all output CSVs, JSONs, and figures from model runs
- `report/`    — final ACL-format PDF report and LaTeX source

## How to Run (Kaggle — free GPU)
1. Upload each notebook to kaggle.com/code
2. Settings → Accelerator → GPU T4 x2 (NB2, NB3, NB4) or None (NB1, NB5)
3. Settings → Internet → On
4. Run All

| Notebook | Description | GPU | Time |
|---|---|---|---|
| NB1_TFIDF | EDA + code-mixing filter + TF-IDF baseline | No | ~5 min |
| NB2_LAFT  | AfroXLMR monolingual x5 languages | Yes | ~90 min |
| NB3_mBERT | mBERT baseline x5 languages | Yes | ~60 min |
| NB4_MAFT  | AfroXLMR joint multilingual | Yes | ~30 min |
| NB5_Results | Merge + error analysis + final table | No | ~10 min |

## Results (Colab run — 22 May 2026, Run ID: 20260521_2237)
| Model | Hausa | Yorùbá | Igbo | Pidgin | Swahili |
|---|---|---|---|---|---|
| TF-IDF+LR | 69.84 | 72.64 | 77.35 | 65.06 | 57.83 |
| mBERT     | 74.25 | 66.05 | 75.78 | 64.33 | 52.67 |
| LAFT      | 76.34 | 70.27 | 77.24 | 68.02 | 60.86 |
| MAFT      | 75.57 | 70.27 | 75.02 | 69.00 | 62.51 |

## Dataset
AfriSenti (Muhammad et al., 2023) loaded automatically from:
https://github.com/afrisenti-semeval/afrisent-semeval-2023
