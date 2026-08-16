import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from preprocess import clean_text, load_dataset  # noqa: E402


def test_clean_text_strips_whitespace():
    assert clean_text("  hello world  ") == "hello world"


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_load_dataset_parses_labels_correctly(tmp_path):
    tsv_path = tmp_path / "sample.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ham", "hey how are you"])
        writer.writerow(["spam", "WIN FREE CASH NOW"])
        writer.writerow(["ham", "see you at 5"])

    texts, labels = load_dataset(tsv_path)

    assert len(texts) == 3
    assert len(labels) == 3
    assert labels == [0, 1, 0]
    assert texts[1] == "WIN FREE CASH NOW"


def test_load_dataset_skips_malformed_rows(tmp_path):
    tsv_path = tmp_path / "sample.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ham", "valid row"])
        writer.writerow(["onlyonecolumn"])
        writer.writerow(["spam", "another valid row"])

    texts, labels = load_dataset(tsv_path)

    assert len(texts) == 2
    assert labels == [0, 1]
