import re
import json
import pandas as pd
from pathlib import Path


def load_offensive_words(path="assets/offensive_words.json"):
    """Load offensive words from a JSON array and normalize to lowercase."""
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return [w.strip().lower() for w in data if isinstance(w, str) and w.strip()]


def mask_offensive(text, offensive_words):
    """Replace offensive words in text with [OFFENSIVE]."""
    if not offensive_words:
        return text
    # Escape special regex chars and match full words (case-insensitive)
    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, offensive_words)) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub("[OFFENSIVE]", text)


def process_tsv(in_path, out_path, offensive_words):
    """Read TSV file, mask offensive words in text column, and write output."""
    df = pd.read_csv(in_path, sep="\t", header=None)
    df[0] = df[0].astype(str).apply(
        lambda t: mask_offensive(t, offensive_words))
    df.to_csv(out_path, sep="\t", header=False, index=False)


def main():
    offensive_words = load_offensive_words()
    Path("data_masked").mkdir(exist_ok=True)

    for split in ["train", "dev", "test"]:
        process_tsv(f"data/{split}.tsv",
                    f"data_masked/{split}.tsv", offensive_words)
        print(f"Masked {split}.tsv -> data_masked/{split}.tsv")


if __name__ == "__main__":
    main()
