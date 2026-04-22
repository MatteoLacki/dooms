<p align="center">
  <img src="pics/logo.png" alt="dooms logo" width="160">
</p>

# dooms

ANOVA and Tukey post-hoc analysis for long-format proteomics data.

Input CSV files must contain:

- `Protein.Group`
- `condition`
- `Intensity`

Run from an installed environment:

```bash
dooms input.csv --output-dir dooms_outputs
```

or without installing a console script:

```bash
python -m dooms input.csv --output-dir dooms_outputs
```

The CLI writes two unfiltered result tables:

- `anova_results.csv`: one row per protein with F-statistic, raw ANOVA p-value, FDR q-value, and group-presence metadata.
- `tukey_results_long.csv`: one row per protein and condition pair with Tukey statistic, raw p-value, and all Tukey FDR q-values.

## Use As A Library

Run the full CSV-to-CSV workflow from Python:

```python
from dooms import run_analysis

result = run_analysis(
    "input.csv",
    "dooms_outputs",
    p_cut=0.01,
    fdr=0.01,
    tukey_sf_backend="numba_ch",
    verbose=False,
)

print(result["anova_path"])
print(result["tukey_path"])
anova_results = result["anova_results"]
tukey_results_long = result["tukey_results_long"]
```

Run directly on an in-memory dataframe:

```python
import pandas as pd
from dooms import anova_test

data = pd.read_csv("input.csv")
anova_results, tukey_wide, tukey_long, timings = anova_test(
    data,
    p_cut=0.01,
    fdr=0.01,
    tukey_sf_backend="numba_ch",
    verbose=False,
    return_tukey_long=True,
)
```

The dataframe must contain `Protein.Group`, `condition`, and `Intensity`.
Non-finite intensities are ignored. If a protein is missing any condition after
that filtering, its ANOVA and Tukey statistics are reported as `NaN`.

Decode group-presence masks:

```python
from dooms import encode_group_presence_mask, decode_group_presence_mask

mask_int = encode_group_presence_mask([True, True, False, False, True, False])
assert mask_int == 19

present = decode_group_presence_mask(mask_int, 6)
print(present.tolist())  # [True, True, False, False, True, False]
```

Condition bit order follows the first-seen order in the input data.
