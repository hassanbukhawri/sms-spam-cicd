"""
Downloads the SMS Spam Collection dataset (ham/spam labeled SMS messages).

We fetch this at pipeline-run time rather than committing the raw file to
the repo. Keeps the repo small and makes the data source explicit and
reproducible (anyone running the pipeline gets the exact same source).
"""

import csv
import sys
from pathlib import Path
from urllib.request import urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sms.tsv"


def fetch(url: str, output_path: Path, timeout: int = 30) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response:
        content = response.read().decode("utf-8")

    output_path.write_text(content, encoding="utf-8")

    # Sanity check: confirm it parses as a 2-column TSV with expected labels
    with output_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    if len(rows) < 1000:
        raise ValueError(
            f"Downloaded file has only {len(rows)} rows, expected 1000+. "
            "Data source may have changed or download was truncated."
        )

    labels = {row[0] for row in rows if row}
    if not labels.issubset({"ham", "spam"}):
        raise ValueError(f"Unexpected labels found: {labels}")

    print(f"Downloaded {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    try:
        fetch(DATA_URL, OUTPUT_PATH)
    except Exception as e:
        print(f"ERROR: failed to fetch dataset: {e}", file=sys.stderr)
        sys.exit(1)
