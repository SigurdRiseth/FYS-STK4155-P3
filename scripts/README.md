# scripts/

Standalone, runnable entry points that orchestrate the package in
`src/fys_stk4155_p0/` — e.g. `download_data.py`, `run_experiment.py`,
`make_figures.py`. Scripts should stay thin: put reusable logic in the
package under `src/` so it's importable and testable, and keep only
argument parsing / wiring here.

After the folder `MachineLearningCVE` is downloaded into `data/raw`, you can merge all its content into a single CSV by running the following script:

```bash
uv run scripts/merge_data.py
```
