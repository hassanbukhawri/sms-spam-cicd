"""
Trains a TF-IDF + Logistic Regression spam classifier and writes:
  - models/model.joblib   (the fitted sklearn Pipeline)
  - models/metrics.json   (held-out test metrics, consumed by evaluate.py)

Logistic Regression over Naive Bayes: NB is the traditional baseline for
spam detection, but it assumes conditional feature independence, which text
data violates. Logistic Regression on TF-IDF features tends to calibrate
better and is what we compare against in evaluate.py's --compare-nb flag.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import load_dataset  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sms.tsv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # spam is a minority class
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train(data_path: Path, model_dir: Path) -> dict:
    texts, labels = load_dataset(data_path)
    if len(texts) < 100:
        raise ValueError(f"Only {len(texts)} rows loaded — dataset looks wrong.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    with (model_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    metrics = train(args.data, args.model_dir)
    print(json.dumps(metrics, indent=2))
