# scripts/

Standalone, runnable entry points that orchestrate the package in
`src/fys_stk4155_p0/` — e.g. `download_data.py`, `run_experiment.py`,
`make_figures.py`. Scripts should stay thin: put reusable logic in the
package under `src/` so it's importable and testable, and keep only
argument parsing / wiring here.

After the folder `MachineLearningCVE` is downloaded into `data/raw`, you can merge and clean all its content by running the following script:

```bash
uv run scripts/prepare_data.py
```

This writes two files to `data/processed/`:

- `cicids2017_raw.parquet`: all CSV files merged, untouched (used by the EDA notebook).
- `cicids2017_clean.parquet`: the output of `fys_stk4155_p3.data.cleaning.clean`.
