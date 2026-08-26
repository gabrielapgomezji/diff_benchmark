# Analysis API

The analysis utilities can also be used directly from Python.

For a conceptual overview of the analysis stage, see [Analysis](../user_guide/analysis.md).

## Building the comprehensive results table

The comprehensive table can be generated programmatically with:

```python
from pathlib import Path

from diff_benchmark.cli.analysis import build_comprehensive_table

df = build_comprehensive_table(
    experiments_root=Path("exp_outputs/experiments"),
    output_path=Path(
        "exp_outputs/summary/comprehensive_table.parquet"
    ),
)

print(df.columns.tolist())
```

The resulting table contains aggregated experiment metrics together with the corresponding experiment configuration.

This is useful when custom analyses need to be built on top of the standard DiffBench outputs.
