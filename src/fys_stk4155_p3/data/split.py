"""Deterministic stratified train/validation/test splitting."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class PartitionReport:
    """Size and class distribution of a single partition."""

    rows: int
    fraction: float
    label_counts: dict[str, int]


@dataclass
class SplitReport:
    """Summary of a split run, intended to be saved and cited in the report."""

    seed: int
    train_size: float
    val_size: float
    test_size: float
    target_column: str
    input_rows: int
    partitions: dict[str, PartitionReport] = field(default_factory=dict)
    # Provenance, filled in by the caller since it depends on the environment.
    git_commit: str = ""
    package_versions: dict[str, str] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Write the report as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    target_column: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stratified train, validation, and test partitions."""
    train, val, test, _ = split_with_report(
        df, train_size, val_size, test_size, seed, target_column
    )
    return train, val, test


def split_with_report(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    target_column: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """
    Create deterministic stratified train, validation, and test partitions.

    Stratification uses the supplied target column. The default is the original
    fine-grained CICIDS2017 Label column. The index must be unique; it is kept
    in each partition so rows can be traced back to the input.

    Returns:
        Tuple containing train, validation, and test DataFrames, and a report.
    """
    sizes = (train_size, val_size, test_size)

    if any(size <= 0 or size >= 1 for size in sizes):
        raise ValueError("All split sizes must be between 0 and 1")
    if not abs(sum(sizes) - 1) < 1e-9:
        raise ValueError("All split sizes summed must total to 1")
    if not df.index.is_unique:
        raise ValueError("DataFrame index must be unique")

    logger.info(
        "Splitting the data with seed %d and size (%f, %f, %f)",
        seed,
        train_size,
        val_size,
        test_size,
    )

    # Split on the index rather than the DataFrame itself, then select rows with
    # .loc. This keeps the results typed as DataFrames (sklearn's stubs type the
    # return value of train_test_split as a list).

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

    report = SplitReport(
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        target_column=target_column,
        input_rows=len(df),
    )
    for name, part in (("train", train), ("val", val), ("test", test)):
        partition = PartitionReport(
            rows=len(part),
            fraction=round(len(part) / len(df), 6),
            label_counts={str(k): int(v) for k, v in part[target_column].value_counts().items()},
        )
        report.partitions[name] = partition
        logger.info(
            "%s: %d rows (%.2f%%)\n%s",
            name,
            partition.rows,
            100 * partition.fraction,
            part[target_column].value_counts().to_string(),
        )

    return train.copy(), val.copy(), test.copy(), report
