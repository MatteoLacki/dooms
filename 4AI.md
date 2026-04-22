# dooms Notes For AI Agents

`dooms` is a Python package for one-way ANOVA plus Tukey post-hoc analysis on
long-format proteomics data. It was extracted from the older root-level
`specloom.py` script.

## Input Contract

The CLI and `dooms.io.run_analysis()` expect a CSV with these columns:

- `Protein.Group`: feature/protein identifier.
- `condition`: group label.
- `Intensity`: numeric measurement.

Rows with non-finite `Intensity` values are ignored during aggregation. A
protein only receives valid ANOVA and Tukey values if every condition has at
least one non-NaN observation after filtering. Otherwise its raw p-values and
statistics are reported as `NaN`.

Condition order matters. Conditions are encoded in first-seen input order, not
alphabetical order. Presence bit `2**0` is the first condition encountered,
`2**1` is the second, etc.

## Outputs

The CLI writes exactly two unfiltered CSV files:

- `anova_results.csv`: one row per protein.
- `tukey_results_long.csv`: one row per protein and condition pair.

No filtered significant-only CSVs should be emitted by the CLI.

ANOVA output includes:

- `group_presence_mask`
- `n_total`
- `n_groups_present`
- `df_between`
- `df_within`
- `F-value `
- `p-value `
- `anova_q_value`
- `anova_q_value_global`
- compatibility columns: `adjusted p-value `, `significant_bool`, `significant`

Tukey long output includes:

- `group_presence_mask`
- `group_a`
- `group_b`
- `comparison`
- `tukey_statistic`
- `tukey_p_value`
- raw-threshold `significant`
- FDR columns:
  - `tukey_q_value_global`
  - `tukey_q_value_by_comparison`
  - `tukey_q_value_by_protein`
  - `tukey_q_value_gated_global`
  - `tukey_q_value_gated_by_comparison`
  - `tukey_q_value_gated_by_protein`
  - corresponding `significant_*` boolean columns

## Package Structure

- `dooms.analysis`: main ANOVA/Tukey implementation and numba kernels.
- `dooms.fdr`: numba-compatible Benjamini-Hochberg helpers.
- `dooms.masks`: group-presence bitmask helpers.
- `dooms.io`: CSV runner that writes the two output tables.
- `dooms.cli.main`: argparse CLI.
- `dooms.__main__`: enables `python -m dooms`.

Public imports from `dooms`:

- `anova_test`
- `run_analysis`
- `encode_group_presence_mask`
- `decode_group_presence_mask`

## CLI

Run with:

```bash
dooms input.csv --output-dir dooms_outputs
```

or:

```bash
python -m dooms input.csv --output-dir dooms_outputs
```

Important CLI options:

- `--tukey-backend`: one of `numba`, `numba_adaptive`, `numba_ch`,
  `numba_grouped`, `numba_grouped_interp`, `scipy_vector`, `scipy_grouped`.
- `--p-cut`: raw p-value cutoff, default `0.01`.
- `--fdr`: q-value cutoff, default `0.01`.
- `-q/--quiet`: suppress core timing output.

Default Tukey backend in the CLI runner is `numba_ch`, because it was a good
accuracy/speed tradeoff on the available data.

## Implementation Notes

- The ANOVA accumulator ignores non-finite intensities.
- ANOVA is intentionally strict about missing groups: all conditions must be
  present for a protein to get non-NaN ANOVA/Tukey statistics.
- Tukey uses Tukey-Kramer style standard errors through harmonic mean pair
  sizes.
- FDR corrections are computed with package-local numba BH helpers, not
  statsmodels.
- `anova_test()` still returns a wide Tukey table in memory for compatibility,
  but `run_analysis()` and the CLI only write the ANOVA table and long Tukey
  table.

## Verification Commands

From `dooms/dooms`:

```bash
make install

make check

make cli-check
```

The Makefile creates a local virtualenv at `ve_dooms` and installs the package
there in editable mode. Do not rely on virtualenvs from parent directories.
