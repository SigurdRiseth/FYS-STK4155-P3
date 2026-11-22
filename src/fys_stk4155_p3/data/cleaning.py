"""Deterministic cleaning utilities for CICIDS2017."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABEL_COLUMN = "Label"
SOURCE_COLUMN = "source_file"

PROTECTED_COLUMNS = {LABEL_COLUMN, SOURCE_COLUMN}

# -1 is a sentinel for "not observed" in these columns, not a measurement artefact.
NEGATIVE_ALLOWED_COLUMNS = {"Init_Win_bytes_forward", "Init_Win_bytes_backward"}

Step = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class StepReport:
    """What a single cleaning step removed."""

    step: str
    rows_before: int
    rows_after: int
    rows_dropped: int
    columns_dropped: list[str]


@dataclass
class CleaningReport:
    """Summary of a cleaning run, intended to be saved and cited in the report."""

    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    memory_mb_before: float = 0.0
    memory_mb_after: float = 0.0
    steps: list[StepReport] = field(default_factory=list)
    label_counts_before: dict[str, int] = field(default_factory=dict)
    label_counts_after: dict[str, int] = field(default_factory=dict)
    attack_family_counts: dict[str, int] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Write the report as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of a CICIDS2017 DataFrame."""
    cleaned, _ = clean_with_report(df)
    return cleaned


def clean_with_report(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a cleaned copy of the DataFrame and a report of what was removed."""
    logger.info("Starting cleaning with %d rows and %d columns", *df.shape)

    report = CleaningReport(
        input_shape=(int(df.shape[0]), int(df.shape[1])),
        memory_mb_before=_memory_mb(df),
    )

    df = _run_steps(df, (clean_column_names, normalize_labels), report)
    report.label_counts_before = _value_counts(df[LABEL_COLUMN])

    df = _run_steps(
        df,
        (
            drop_constant_columns,
            drop_duplicate_columns,
            replace_infinite_values,
            drop_missing_rows,
            drop_negative_rows,
            # Downcast before deduplicating: rows that differ only beyond float32
            # precision become identical after the downcast.
            downcast_numeric_columns,
            drop_conflicting_duplicates,
            drop_duplicate_rows,
            add_targets,
        ),
        report,
    )

    report.output_shape = (int(df.shape[0]), int(df.shape[1]))
    report.memory_mb_after = _memory_mb(df)
    report.label_counts_after = _value_counts(df[LABEL_COLUMN])
    report.attack_family_counts = _value_counts(df["attack_family"])

    removed = report.input_shape[0] - report.output_shape[0]
    logger.info(
        "Cleaning complete: %d -> %d rows (removed %d)",
        report.input_shape[0],
        report.output_shape[0],
        removed,
    )

    return df, report


def _run_steps(df: pd.DataFrame, steps: tuple[Step, ...], report: CleaningReport) -> pd.DataFrame:
    """Apply each step in order and record what it removed."""
    for step in steps:
        rows_before, columns_before = len(df), list(df.columns)
        df = step(df)
        report.steps.append(
            StepReport(
                step=step.__name__,
                rows_before=rows_before,
                rows_after=len(df),
                rows_dropped=rows_before - len(df),
                # Stripped names count as renamed, not dropped.
                columns_dropped=[c for c in columns_before if str(c).strip() not in df.columns],
            )
        )
    return df


def _memory_mb(df: pd.DataFrame) -> float:
    return round(float(df.memory_usage(deep=True).sum()) / 1e6, 1)


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespaces from column names and reject name collisions."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    duplicates = df.columns[df.columns.duplicated()].to_list()
    if duplicates:
        raise ValueError(f"Duplicate column names after stripping: {duplicates}")

    return df


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known CICIDS2017 label encoding artefacts."""
    df = df.copy()

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {LABEL_COLUMN}")

    labels = df[LABEL_COLUMN].astype(str).str.strip()

    labels = labels.str.replace("�", "-", regex=False)
    labels = labels.str.replace(r"\s+", " ", regex=True)

    df[LABEL_COLUMN] = labels
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns containing at most one distinct non-null value."""
    df = df.copy()

    candidates = [column for column in df.columns if column not in PROTECTED_COLUMNS]

    constant = [column for column in candidates if _is_constant(df[column])]

    if constant:
        logger.info("Dropping %d constant columns: %s", len(constant), constant)
        df = df.drop(columns=constant)

    return df


def _is_constant(series: pd.Series) -> bool:
    """Return True if the series has at most one distinct non-null value."""
    values = series.dropna().to_numpy()
    return values.size == 0 or bool((values == values[0]).all())


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with identical values, preserving protected columns."""
    df = df.copy()
    keep: list[str] = []
    dropped: list[str] = []

    for column in df.columns:
        if column in PROTECTED_COLUMNS:
            keep.append(column)
            continue

        duplicate_of = next(
            (
                previous
                for previous in keep
                if previous not in PROTECTED_COLUMNS and df[column].equals(df[previous])
            ),
            None,
        )

        if duplicate_of is None:
            keep.append(column)
        else:
            dropped.append(column)
            logger.info(
                "Dropping duplicate column %r (identical to %r)",
                column,
                duplicate_of,
            )

    if dropped:
        df = df.drop(columns=dropped)

    return df


def replace_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace positive and negative infinity with NaN."""
    df = df.copy()

    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values."""
    before = len(df)
    df = df.dropna()
    logger.info("Dropped %d rows containing NaN values", before - len(df))
    return df


def drop_negative_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with negative values (measurement artefacts), except sentinel columns."""
    numeric_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in NEGATIVE_ALLOWED_COLUMNS
    ]

    negative = df[numeric_columns] < 0
    counts = negative.sum()
    counts = counts[counts > 0]
    if not counts.empty:
        logger.info("Negative values per column: %s", counts.to_dict())

    invalid = negative.any(axis=1)
    logger.info("Dropped %d rows with negative values", int(invalid.sum()))
    return df.loc[~invalid].copy()


def downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 to float32 and int64 to int32 where the values fit."""
    df = df.copy()

    for column in df.select_dtypes(include="float64").columns:
        df[column] = df[column].astype("float32")

    int32 = np.iinfo(np.int32)
    for column in df.select_dtypes(include="int64").columns:
        if df[column].min() >= int32.min and df[column].max() <= int32.max:
            df[column] = df[column].astype("int32")

    return df


def drop_conflicting_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove feature-identical rows with conflicting labels.

    Source metadata is excluded from the comparison so that the same
    observation appearing in different source files is still detected.
    """
    df = df.copy()

    feature_columns = [
        column for column in df.columns if column not in {LABEL_COLUMN, SOURCE_COLUMN}
    ]

    duplicated = df.duplicated(subset=feature_columns, keep=False)

    if not duplicated.any():
        return df

    duplicate_groups = (
        df.loc[duplicated]
        .groupby(
            feature_columns,
            dropna=False,
            sort=False,
        )[LABEL_COLUMN]
        .nunique()
    )

    conflicting_groups = duplicate_groups[duplicate_groups > 1]

    if conflicting_groups.empty:
        return df

    # Identify all rows belonging to a conflicting feature group.
    conflict_index = df.set_index(feature_columns).index.isin(conflicting_groups.index)

    removed = int(conflict_index.sum())
    logger.warning(
        "Dropping %d rows belonging to conflicting-label duplicate groups",
        removed,
    )

    return df.loc[~conflict_index].copy()


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate observations, ignoring source metadata."""
    df = df.copy()

    subset = [column for column in df.columns if column != SOURCE_COLUMN]

    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")

    logger.info("Dropped %d duplicate rows", before - len(df))
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary and grouped multiclass targets."""
    df = df.copy()

    labels = df[LABEL_COLUMN].astype("string")

    df["is_attack"] = labels.ne("BENIGN").astype("int8")

    def get_attack_family(label: str) -> str:
        if label == "BENIGN":
            return "BENIGN"
        if label == "DDoS":
            return "DDoS"
        if label.startswith("DoS "):
            return "DoS"
        if label == "PortScan":
            return "PortScan"
        if label in {"FTP-Patator", "SSH-Patator"}:
            return "Brute Force"
        if label.startswith("Web Attack"):
            return "Web Attack"
        if label == "Bot":
            return "Bot"
        if label in {"Infiltration", "Heartbleed"}:
            return "Rare"

        raise ValueError(f"Unrecognized CICIDS2017 label: {label!r}")

    df["attack_family"] = labels.map(get_attack_family)

    return df
