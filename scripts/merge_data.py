from pathlib import Path

from fys_stk4155_p3.data.merging import merge_datasets


def main() -> None:
    data_dir = Path("data/raw/machine-learning-cve")
    output_path = Path("data/processed/cicids2017.parquet")

    df = merge_datasets(data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
