import pandas as pd
import pytest

from fys_stk4155_p3.data.split import split, split_with_report


@pytest.fixture
def df() -> pd.DataFrame:
    """Imbalanced frame with a non-contiguous index, as left behind by cleaning."""
    labels = ["BENIGN"] * 800 + ["DoS"] * 180 + ["Rare"] * 20
    return pd.DataFrame(
        {"x": range(len(labels)), "Label": labels},
        index=pd.RangeIndex(0, 2 * len(labels), 2, name="row_id"),
    )


def test_partitions_are_disjoint_and_complete(df: pd.DataFrame) -> None:
    train, val, test = split(df)

    assert (len(train), len(val), len(test)) == (700, 150, 150)
    assert sorted([*train.index, *val.index, *test.index]) == list(df.index)
    assert train.index.name == "row_id"


def test_stratifies_on_label(df: pd.DataFrame) -> None:
    _, _, test = split(df)

    assert test["Label"].value_counts().to_dict() == {"BENIGN": 120, "DoS": 27, "Rare": 3}


def test_is_deterministic(df: pd.DataFrame) -> None:
    first, second = split(df), split(df)
    other_seed = split(df, seed=0)

    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)
    assert not first[2].index.equals(other_seed[2].index)


def test_report_matches_partitions(df: pd.DataFrame) -> None:
    train, _, test, report = split_with_report(df)

    assert report.input_rows == len(df)
    assert report.partitions["train"].rows == len(train)
    assert report.partitions["test"].label_counts == test["Label"].value_counts().to_dict()


@pytest.mark.parametrize(
    "sizes",
    [(0.7, 0.2, 0.2), (1.0, 0.0, 0.0), (0.8, -0.1, 0.3)],
)
def test_rejects_invalid_sizes(df: pd.DataFrame, sizes: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="split sizes"):
        split(df, *sizes)


def test_rejects_duplicate_index(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unique"):
        split(pd.concat([df, df]))
