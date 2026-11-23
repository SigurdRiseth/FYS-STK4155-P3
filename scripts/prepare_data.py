import argparse
import logging
import subprocess
from importlib.metadata import version
from pathlib import Path

from fys_stk4155_p3.data.cleaning import clean_with_report
from fys_stk4155_p3.data.merging import merge_datasets
from fys_stk4155_p3.data.split import split_with_report

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/raw/machine-learning-cve"
OUTPUT_DIR = ROOT / "data/processed"
RESULTS_DIR = ROOT / "results"

RAW_PATH = OUTPUT_DIR / "cicids2017_raw.parquet"
CLEANED_PATH = OUTPUT_DIR / "cicids2017_clean.parquet"
SPLIT_PATHS = {name: OUTPUT_DIR / f"{name}.parquet" for name in ("train", "val", "test")}
CLEANING_REPORT_PATH = RESULTS_DIR / "cleaning_report.json"
SPLIT_REPORT_PATH = RESULTS_DIR / "split_report.json"

# Index name linking cleaned rows and splits back to rows of the raw parquet.
ROW_ID = "row_id"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge, clean, and split CICIDS2017.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing train/val/test splits. This changes the test set.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Check before doing any work: regenerating the splits must be deliberate.
    existing = [path for path in SPLIT_PATHS.values() if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            f"Splits already exist ({', '.join(p.name for p in existing)}). "
            "Re-run with --force to regenerate them."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = merge_datasets(DATA_DIR)
    df.index.name = ROW_ID
    df.to_parquet(RAW_PATH)

    # Clean up and write the cleaned data, plus a report of what was removed.
    # Cleaning drops rows without resetting the index, so row_id is preserved.
    df, cleaning_report = clean_with_report(df)
    df.to_parquet(CLEANED_PATH)
    cleaning_report.to_json(CLEANING_REPORT_PATH)
    logging.info("Wrote cleaning report to %s", CLEANING_REPORT_PATH)

    # Create and write stratified splits, plus a report of their contents
    train, val, test, split_report = split_with_report(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_parquet(SPLIT_PATHS[name])
    logging.info("Wrote train, validation, and test splits to %s", OUTPUT_DIR)

    split_report.git_commit = _git_commit()
    split_report.package_versions = {
        package: version(package) for package in ("numpy", "pandas", "scikit-learn")
    }
    split_report.to_json(SPLIT_REPORT_PATH)
    logging.info("Wrote split report to %s", SPLIT_REPORT_PATH)


def _git_commit() -> str:
    """Return the current commit hash, suffixed with -dirty if there are local changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


if __name__ == "__main__":
    main()
