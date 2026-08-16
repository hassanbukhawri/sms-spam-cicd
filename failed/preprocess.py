"""
Shared text preprocessing utilities. Kept deliberately simple (lowercase +
strip) because TF-IDF handles tokenization/weighting downstream, and the
sklearn Pipeline bundles the vectorizer with the model so preprocessing is
never inconsistent between train and serve time.
"""

import csv
from pathlib import Path
from typing import List, Tuple


def load_dataset(path: Path) -> Tuple[List[str], List[int]]:
    """Loads the TSV dataset and returns (texts, labels) with labels as 0/1."""
    texts, labels = [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            label, text = row[0], row[1]
            texts.append(clean_text(text))
            labels.append(1 if label == "spam" else 0)
    return texts, labels


def clean_text(text: str) -> str:
    """Minimal normalization. TF-IDF's tokenizer + stopword handling does
    the heavy lifting, so we avoid over-engineering this step."""
    return text.strip()
