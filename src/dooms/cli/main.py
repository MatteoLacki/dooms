from __future__ import annotations

import argparse
from pathlib import Path

from dooms.io import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dooms",
        description="Run ANOVA and Tukey post-hoc analysis for long-format proteomics data.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input CSV with Protein.Group, condition, and Intensity columns.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("dooms_outputs"),
        help="Directory for result CSV files.",
    )
    parser.add_argument("--p-cut", type=float, default=0.01, help="Raw p-value cutoff.")
    parser.add_argument("--fdr", type=float, default=0.01, help="FDR q-value cutoff.")
    parser.add_argument(
        "--tukey-backend",
        default="numba_ch",
        choices=[
            "numba",
            "numba_adaptive",
            "numba_ch",
            "numba_grouped",
            "numba_grouped_interp",
            "scipy_vector",
            "scipy_grouped",
        ],
        help="Studentized range survival-function backend.",
    )
    parser.add_argument(
        "--anova-filename",
        default="anova_results.csv",
        help="Output filename for the ANOVA/F-stat result table.",
    )
    parser.add_argument(
        "--tukey-filename",
        default="tukey_results_long.csv",
        help="Output filename for the long-format Tukey result table.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress analysis timing output from the computational core.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_analysis(
        args.input,
        args.output_dir,
        p_cut=args.p_cut,
        fdr=args.fdr,
        tukey_sf_backend=args.tukey_backend,
        verbose=not args.quiet,
        anova_filename=args.anova_filename,
        tukey_filename=args.tukey_filename,
    )
    print(f"ANOVA results: {result['anova_path']}")
    print(f"Tukey results: {result['tukey_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
