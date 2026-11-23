import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    target_column: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create deterministic stratified train, validation, and test partitions.

    Stratification uses the supplied target column. The default is the original
    fine-grained CICIDS2017 Label column.

    Returns:
        Tuple containing train, validation, and test DataFrames.
    """
    sizes = (train_size, val_size, test_size)

    if any(size <= 0 or size >= 1 for size in sizes):
        raise ValueError("All split sizes must be between 0 and 1")
    if not abs(sum(sizes) - 1) < 1e-9:
        raise ValueError("All split sizes summed must total to 1")

    logger.info(
        "Splitting the data with seed %d and size (%f, %f, %f)",
        seed,
        train_size,
        val_size,
        test_size,
    )

    # First reserve the test partition
    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_column],
    )
    train_val = df.loc[train_val_idx]
    test = df.loc[test_idx]

    # Split the remaining data into train and validation
    relative_val_size = val_size / (train_size + val_size)

    train_idx, val_idx = train_test_split(
        train_val.index,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val[target_column],
    )
    train = train_val.loc[train_idx]
    val = train_val.loc[val_idx]

    total = len(df)
    _log_split("Train", train, total, target_column)
    _log_split("Validation", val, total, target_column)
    _log_split("Test", test, total, target_column)

    return train.copy(), val.copy(), test.copy()


def _log_split(
    name: str,
    df: pd.DataFrame,
    total: int,
    target_column: str,
) -> None:
    """Log partition size, share of the full dataset, and class distribution."""
    logger.info(
        "%s: %d rows (%.2f%%)",
        name,
        len(df),
        100 * len(df) / total,
    )
    logger.info(
        "%s class counts:\n%s",
        name,
        df[target_column].value_counts().to_string(),
    )
