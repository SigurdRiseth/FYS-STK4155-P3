"""Loading the processed splits and separating features from targets."""

from pathlib import Path
from typing import Literal

import pandas as pd

from fys_stk4155_p3.data.cleaning import LABEL_COLUMN, SOURCE_COLUMN

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data/processed"

Task = Literal["binary", "family", "label"]

TARGET_COLUMNS: dict[Task, str] = {
    "binary": "is_attack",
    "family": "attack_family",
    "label": LABEL_COLUMN,
}

# Columns that are not features: every target (so none leaks into X) and metadata.
NON_FEATURE_COLUMNS = {*TARGET_COLUMNS.values(), SOURCE_COLUMN}


def load_splits(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train, validation, and test parquet files written by prepare_data.py."""
    train, val, test = (
        pd.read_parquet(data_dir / f"{name}.parquet") for name in ("train", "val", "test")
    )
    return train, val, test


def get_xy(df: pd.DataFrame, task: Task) -> tuple[pd.DataFrame, pd.Series]:
    """Split a partition into the feature matrix and the target for the given task."""
    features = [column for column in df.columns if column not in NON_FEATURE_COLUMNS]
    return df[features], df[TARGET_COLUMNS[task]]
