import pandas as pd
import pytest

from fys_stk4155_p3.data.load import get_xy


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow Duration": [10, 20],
            "Flow IAT Min": [1, 2],
            "Label": ["BENIGN", "DoS Hulk"],
            "source_file": ["a.csv", "b.csv"],
            "is_attack": [0, 1],
            "attack_family": ["BENIGN", "DoS"],
        }
    )


@pytest.mark.parametrize(
    ("task", "target"),
    [("binary", "is_attack"), ("family", "attack_family"), ("label", "Label")],
)
def test_get_xy_excludes_all_targets_and_metadata(df: pd.DataFrame, task, target) -> None:
    X, y = get_xy(df, task)

    assert list(X.columns) == ["Flow Duration", "Flow IAT Min"]
    assert y.name == target
