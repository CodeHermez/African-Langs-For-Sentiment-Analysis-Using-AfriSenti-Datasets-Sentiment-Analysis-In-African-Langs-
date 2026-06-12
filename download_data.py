"""
download_data.py — Group 60 AfriSenti Data Download Script
===========================================================
Downloads the AfriSenti-SemEval 2023 dataset for all 5 languages
(Hausa, Yoruba, Igbo, Nigerian Pidgin, Swahili) from the official
GitHub repository and saves each split as a .tsv file locally.

Usage:
    python download_data.py

Output structure:
    data/
    ├── hau/
    │   ├── train.tsv
    │   ├── dev.tsv
    │   └── test.tsv
    ├── yor/
    ├── ibo/
    ├── pcm/
    └── swa/

Requirements:
    pip install pandas
"""

import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
BASE_URL = (
    "https://raw.githubusercontent.com/afrisenti-semeval/"
    "afrisent-semeval-2023/main/data/{lang}/{split}.tsv"
)

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}

LANGUAGES = {
    "hau": "Hausa",
    "yor": "Yoruba",
    "ibo": "Igbo",
    "pcm": "Nigerian Pidgin",
    "swa": "Swahili",
}

SPLITS = ["train", "dev", "test"]

OUTPUT_DIR = Path("data")


# ── Functions ─────────────────────────────────────────────────────
def download_split(lang_iso: str, split: str) -> pd.DataFrame:
    url = BASE_URL.format(lang=lang_iso, split=split)
    df = pd.read_csv(
        url, sep="\t", header=None, names=["tweet", "label"], on_bad_lines="skip"
    )
    df = df.dropna(subset=["tweet", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(LABEL2ID)].copy()
    df["label_id"] = df["label"].map(LABEL2ID)
    df["iso"] = lang_iso
    return df.reset_index(drop=True)


def download_language(iso: str):
    print(f"\n  Downloading {LANGUAGES[iso]}...")
    lang_dir = OUTPUT_DIR / iso
    lang_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        df = download_split(iso, split)
        out_path = lang_dir / f"{split}.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        print(f"    {split}: {len(df)} rows → saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("AfriSenti-SemEval 2023 — Data Download Script")
    print("=" * 50)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for iso in LANGUAGES:
        download_language(iso)

    print("\n✅ All 5 languages downloaded successfully.")
    print(f"   Data saved to: {OUTPUT_DIR.resolve()}")
