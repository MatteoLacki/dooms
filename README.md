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
