# Analysis

> **Pipeline position:** Step 4 — runs after one or more training experiments have completed.  
> **CLI:** `diffbenchmark-analysis`  
> **← [Back to index](index.md)**

---

## Overview

The analysis component post-processes completed experiment directories to produce aggregated metric tables, per-dataset comparison reports, coverage summaries, learning curve plots, and per-run visualisations.

**Inputs:**
- `exp_outputs/experiments/` — directory containing `exp_<run_id>/` sub-directories

**Outputs** (all written to `exp_outputs/`):

| File | Description |
|---|---|
| `summary/global_metrics.parquet` | All fold-level metrics concatenated |
| `summary/summary_metrics.parquet` | Cross-fold mean and std per experiment |
| `summary/comprehensive_table.parquet` | One row per experiment; all metrics and flattened config columns |
| `summary/<dataset>_report.txt` | Per-dataset text report with best-run highlighting |
| `summary/coverage_table.txt` | Coverage matrix across datasets, models, and metrics |
| `plots/<run_id>/` | Per-run visualisations |

---

## Configuration

Analysis behaviour is controlled via the `analysis` Hydra config group and CLI overrides.

| Parameter | Default | Description |
|---|---|---|
| `plots` | `true` | Generate per-run visualisation plots |
| `tables` | `true` | Build and save aggregated metric tables |
| `force_plots` | `false` | Regenerate plots even if they already exist |
| `analysis.debug` | `false` | Include in-progress/failed runs in debug plot generation |

The analysis command reads from `exp_outputs/experiments/` by default; this path is configured in `configs/paths/`.

---

## Usage Example

```bash
# Run full analysis (tables + plots)
diffbenchmark-analysis

# Tables only
diffbenchmark-analysis plots=false

# Regenerate all plots
diffbenchmark-analysis force_plots=true tables=false

# Include debug plots for running/crashed experiments
diffbenchmark-analysis analysis.debug=true
```

```python
# Build comprehensive table programmatically
from pathlib import Path
from diff_benchmark.cli.analysis import build_comprehensive_table

df = build_comprehensive_table(
    experiments_root=Path("exp_outputs/experiments"),
    output_path=Path("exp_outputs/summary/comprehensive_table.parquet"),
)
print(df.columns.tolist())
```
---

## Extra: Reports legend

### Coverage Table Cell Format

Coverage cells encode the tissue type and target as a two-character code, e.g. `gg` = gray matter + gender, `wa` = white matter + age.

```
First letter  : tissue (g = gray, w = white)
Second letter : target (g = gender, a = age, d = diagnosis)
```
--