import logging
from pathlib import Path

from fys_stk4155_p3.data.cleaning import clean_with_report
from fys_stk4155_p3.data.merging import merge_datasets


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_dir = Path("data/raw/machine-learning-cve")
    output_dir = Path("data/processed")
    raw_path = output_dir / "cicids2017_raw.parquet"
    cleaned_path = output_dir / "cicids2017_clean.parquet"
    report_path = Path("results/cleaning_report.json")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read and write raw merged data
    df = merge_datasets(data_dir)
    df.to_parquet(raw_path, index=False)

    # Clean up and write the cleaned data, plus a report of what was removed
    df, report = clean_with_report(df)
    df.to_parquet(cleaned_path, index=False)
    report.to_json(report_path)
    logging.info("Wrote cleaning report to %s", report_path)


if __name__ == "__main__":
    main()
