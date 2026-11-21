from pathlib import Path

import pandas as pd


def merge_datasets(data_dir: Path) -> pd.DataFrame:
    """Merge all CSV files in the given directory into a single dataframe."""

    csv_files = sorted(data_dir.glob("*.csv"))
    return pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
