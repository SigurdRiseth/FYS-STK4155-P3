from pathlib import Path

import pandas as pd


def merge_datasets(data_dir: Path) -> pd.DataFrame:
    """Merge all CSV files in the given directory into a single dataframe."""

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. Download the MachineLearningCVE CSVs "
            "into this folder first (see data/README.md)."
        )

    return pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
