import numpy as np
import pandas as pd
import pytest

from fys_stk4155_p3.data.cleaning import clean_with_report


@pytest.fixture
def raw() -> pd.DataFrame:
    """Small raw-like frame with one example of each problem the cleaning handles."""
    return pd.DataFrame(
        {
            " Flow Duration": [10, 10, 20, 30, 30, 45, -5, 50, 40, 40],
            " Flow IAT Min": [1, 1, 2, 3, 3, 5, 6, -1, 4, 4],
            " Init_Win_bytes_forward": [-1, -1, 100, 100, 100, 100, 100, 100, 100, 100],
            "Flow Bytes/s": [1.0, 1.0, 2.0, 3.0, 3.0, np.inf, 6.0, 7.0, 666666.6666667, 666666.67],
            " Fwd Header Length": [20, 20, 40, 60, 60, 90, 100, 110, 80, 80],
            " Fwd Header Length.1": [20, 20, 40, 60, 60, 90, 100, 110, 80, 80],
            " Bwd PSH Flags": [0] * 10,
            " Label": [
                "BENIGN",  # kept, with the -1 sentinel
                "BENIGN",  # exact duplicate of row 0
                "DoS Hulk",  # kept
                "PortScan",  # conflicts with row 4
                "BENIGN",  # conflicts with row 3
                "BENIGN",  # infinite value
                "BENIGN",  # negative Flow Duration
                "BENIGN",  # negative Flow IAT Min
                "Web Attack � XSS",  # kept, broken label encoding
                "Web Attack � XSS",  # duplicate of row 8 after float32 downcast
            ],
        }
    )


def test_drops_constant_and_duplicate_columns(raw: pd.DataFrame) -> None:
    df, report = clean_with_report(raw)

    assert "Bwd PSH Flags" not in df.columns
    assert "Fwd Header Length.1" not in df.columns
    dropped = {s.step: s.columns_dropped for s in report.steps}
    assert dropped["clean_column_names"] == []  # renamed, not dropped
    assert dropped["drop_constant_columns"] == ["Bwd PSH Flags"]
    assert dropped["drop_duplicate_columns"] == ["Fwd Header Length.1"]


def test_keeps_only_valid_unique_rows(raw: pd.DataFrame) -> None:
    df, _ = clean_with_report(raw)

    assert df["Label"].tolist() == ["BENIGN", "DoS Hulk", "Web Attack - XSS"]
    assert not df.duplicated().any()
    assert (df["Init_Win_bytes_forward"] == -1).any()  # sentinel is kept


def test_targets_and_dtypes(raw: pd.DataFrame) -> None:
    df, _ = clean_with_report(raw)

    assert df["is_attack"].tolist() == [0, 1, 1]
    assert df["attack_family"].tolist() == ["BENIGN", "DoS", "Web Attack"]
    assert df["is_attack"].dtype == np.int8
    assert df["Flow Duration"].dtype == np.int32
    assert df["Flow Bytes/s"].dtype == np.float32


def test_report_row_counts(raw: pd.DataFrame) -> None:
    _, report = clean_with_report(raw)

    dropped = {s.step: s.rows_dropped for s in report.steps}
    assert dropped["drop_missing_rows"] == 1
    assert dropped["drop_negative_rows"] == 2
    assert dropped["drop_conflicting_duplicates"] == 2
    assert dropped["drop_duplicate_rows"] == 2
    assert report.input_shape[0] - report.output_shape[0] == 7
    assert report.label_counts_after == {"BENIGN": 1, "DoS Hulk": 1, "Web Attack - XSS": 1}
