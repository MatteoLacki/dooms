from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import anova_test


def run_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    p_cut: float = 0.01,
    fdr: float = 0.01,
    tukey_sf_backend: str = "numba_ch",
    verbose: bool = True,
    anova_filename: str = "anova_results.csv",
    tukey_filename: str = "tukey_results_long.csv",
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    anova_results, _, tukey_long, timings = anova_test(
        data,
        p_cut=p_cut,
        fdr=fdr,
        tukey_sf_backend=tukey_sf_backend,
        verbose=verbose,
        return_tukey_long=True,
    )

    anova_path = output_dir / anova_filename
    tukey_path = output_dir / tukey_filename
    anova_results.to_csv(anova_path, index=False)
    tukey_long.to_csv(tukey_path, index=False)

    return {
        "anova_path": anova_path,
        "tukey_path": tukey_path,
        "anova_results": anova_results,
        "tukey_results_long": tukey_long,
        "timings": timings,
    }
