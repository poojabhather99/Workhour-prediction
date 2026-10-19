"""Data loading and validation."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "effort.csv"
TARGET = "Effort"
DROP_COLUMNS = ["id", "Project"]
MISSING_MARKERS = ["?", "-", "n/a"]


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the effort dataset, normalising missing-value markers and dropping ID columns."""
    df = pd.read_csv(path, na_values=MISSING_MARKERS)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {path}")
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]
