# Group 60 — Sentiment Analysis in African Languages
## COS760 2025 | AfriSenti NLP Project

**Members:**
- Praises Obi (u26819661)
- Bob Dlamini (u21739120)
- Ishe Allen Chihobo (u22592238)

---

## Overview

This project performs multilingual sentiment analysis across five African languages — **Hausa, Yoruba, Igbo, Nigerian Pidgin, and Swahili** — using the AfriSenti-SemEval 2023 dataset. We implement and compare four model architectures:

| Model | Description |
|---|---|
| **TF-IDF + Logistic Regression** | Baseline model using n-gram features |
| **LAFT** | Language-Adaptive Fine-Tuning of AfroXLMR (monolingual, per language) |
| **mBERT** | Multilingual BERT fine-tuned per language |
| **MAFT** | Multilingual Adaptive Fine-Tuning of AfroXLMR (joint training across all 5 languages) |

Phase 4 extends the analysis with LIME interpretability, code-mixing diagnostics, and an error taxonomy based on Muhammad et al. (2023).

---

## Contents of the Zip File

```
Group60/
├── README.md                    ← This file
├── group60-notebook.ipynb       ← Main project notebook (all 4 phases)
├── download_data.py             ← Standalone script to download the AfriSenti dataset
└── requirements.txt             ← Python library dependencies
```

---

## Setup Instructions

### Requirements
- Python 3.10+
- CUDA-compatible GPU strongly recommended (notebook was developed on Kaggle with a T4 GPU)
- Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** The notebook also includes `!pip install` cells at the top of each phase that handle installation automatically when run in Kaggle.

---

## Running the Code

This project is designed to run as a **Kaggle notebook**. Follow these steps:

### Step 1 — Upload the notebook to Kaggle
1. Go to [kaggle.com](https://www.kaggle.com) and sign in
2. Click **Code → New Notebook**
3. Go to **File → Import Notebook** and upload `group60-notebook.ipynb`

### Step 2 — Attach the pre-computed assets dataset
The model checkpoints, cached results (`all_results.json`), LIME cache, and output plots are hosted as a **public Kaggle dataset**:

🔗 **https://www.kaggle.com/datasets/praisesobiuwc/group60**

To attach it:
1. In the notebook editor, click **+ Add Input** in the right panel
2. Search for `praisesobiuwc/group60`
3. Click **Add** — it will mount automatically at:
   ```
   /kaggle/input/datasets/praisesobiuwc/group60/group60_nlp/
   ```

### Step 3 — Enable GPU and Internet
In the right-hand Settings panel:
- Set **Accelerator** to `GPU T4 x2` (or any available GPU)
- Enable **Internet** (required for data download and model weights)

### Step 4 — Run the notebook

> ⚠️ **Important:** Run the **pip install cell first on its own** before clicking Run All. The notebook's first code cell installs dependencies — if it is included in a Run All, subsequent cells may fail before packages are available.

**Recommended execution order:**
1. Run **Cell 1** (pip install) individually and wait for it to complete
2. Then click **Run All** for the remainder

### What runs vs what is cached

| Component | Behaviour |
|---|---|
| pip installs | Always runs |
| Imports | Always runs |
| Data download (from GitHub) | Always runs |
| Code-mixing filter | Always runs |
| Summary / print cells | Always runs |
| Phase 4 LIME analysis | Always runs |
| TF-IDF training | **Skipped if cached** in `all_results.json` |
| LAFT training | **Skipped if cached** in `all_results.json` |
| mBERT training | **Skipped if cached** in `all_results.json` |
| MAFT training | **Skipped if cached** in `all_results.json` |

All model training results are pre-cached in `all_results.json` inside the attached dataset, so the notebook will skip retraining and load results directly. This means a full run completes in minutes rather than hours.

---

## Data Information

The dataset is the **AfriSenti-SemEval 2023** corpus, downloaded automatically from the official GitHub repository during notebook execution:

🔗 **https://github.com/afrisenti-semeval/afrisent-semeval-2023**

No manual download is required inside Kaggle — the notebook fetches all splits automatically at runtime.

> ⚠️ **For full reproducibility of all results, figures, and plots, use the Kaggle method described above.** This is the only way to access the pre-trained model checkpoints and cached outputs.

A standalone script `download_data.py` is included for anyone who wants to download the raw dataset locally outside of Kaggle. Note that this script only downloads the raw language splits — it does **not** reproduce model training, LIME analysis, or any plots:

```bash
python download_data.py
```

This will create a `data/` folder structured as:

```
data/
├── hau/  (train.tsv, dev.tsv, test.tsv)
├── yor/
├── ibo/
├── pcm/
└── swa/
```

| Language | ISO | Train | Val | Test |
|---|---|---|---|---|
| Hausa | hau | 14,172 | 2,677 | 5,303 |
| Yoruba | yor | 8,522 | 2,090 | 4,515 |
| Igbo | ibo | 10,192 | 1,841 | 3,682 |
| Nigerian Pidgin | pcm | 5,121 | 1,281 | 4,154 |
| Swahili | swa | 1,810 | 453 | 748 |

---

## Model & Hyperparameters

| Parameter | Value |
|---|---|
| Base model (LAFT/MAFT) | `Davlan/afro-xlmr-base` |
| Base model (mBERT) | `bert-base-multilingual-cased` |
| Max sequence length | 128 |
| Batch size (LAFT/MAFT) | 16 (effective 32 with grad accumulation) |
| Batch size (mBERT) | 32 |
| Epochs | 10 (with early stopping, patience=3) |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Seeds | 42, 123, 456 |

---

## Software Versions

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | ≥ 2.2.0 |
| transformers | ≥ 4.38.0 |
| datasets | ≥ 2.18.0 |
| scikit-learn | ≥ 1.4.0 |
| pandas | ≥ 2.2.0 |
| numpy | ≥ 1.26.0 |
| lime | ≥ 0.2.0.1 |
| langdetect | ≥ 1.0.9 |
| matplotlib | ≥ 3.8.0 |
| seaborn | ≥ 0.13.0 |
| accelerate | ≥ 0.27.0 |
