from typing import Tuple, List
import pandas as pd


def load_splits(train_path: str, dev_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/dev/test TSV files that do not contain header rows.

    Each file must have exactly two columns:
      - column 0: the tweet text
      - column 1: the binary label (0 = NOT, 1 = OFF)
    """
    df_tr = pd.read_csv(train_path, sep="\t", header=None)
    df_dv = pd.read_csv(dev_path, sep="\t", header=None)
    df_te = pd.read_csv(test_path, sep="\t", header=None)

    for name, df in [("train", df_tr), ("dev", df_dv), ("test", df_te)]:
        if df.shape[1] < 2:
            raise ValueError(
                f"{name}: expected at least 2 columns (text, label), found {df.shape[1]}")
        if df[1].isnull().any():
            raise ValueError(f"{name}: label column contains null values")

    return df_tr, df_dv, df_te


LABEL_MAP = {"NOT": 0, "OFF": 1}


def map_labels_to_int(series: pd.Series) -> pd.Series:
    """Convert label strings ('NOT' / 'OFF') to integers (0 / 1) with validation."""
    norm = series.astype(str).str.strip().str.upper()

    # Select rows that are NOT in the allowed label map
    invalid_mask = norm.isin(LABEL_MAP.keys()) == False
    if invalid_mask.any():
        examples = norm[invalid_mask].unique()
        raise ValueError(
            f"Found unknown label values: {examples.tolist()} (expected one of {list(LABEL_MAP.keys())})"
        )

    return norm.map(LABEL_MAP)


def to_xy(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """Extract text and labels from a DataFrame with unnamed columns.

    Uses column 0 for text and column 1 for labels.
    """
    X = df[0].astype(str).tolist()
    y = map_labels_to_int(df[1]).astype(int).tolist()
    return X, y
