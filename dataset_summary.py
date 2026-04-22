"""Dataset summary utilities for EMS annotation CSV files.

Edit the parameters in the CONFIG section below and run:

    python dataset_summary.py

The script tolerates annotation files that do not contain an ``IMPACT``
column (e.g. the ``csq`` output), in which case impact-related summaries
are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG — edit these parameters as needed
# --------------------------------------------------------------------------- #
CSV_PATH = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\EMS_annotation\datasets\c_elegans"
    r"\c_elegans.csq.EMS_annotation.csv"
)
STRAIN_COLUMN = "STRAIN"
CONSEQUENCE_COLUMN = "CONSEQUENCE"
IMPACT_COLUMN = "IMPACT"
# --------------------------------------------------------------------------- #


def load_annotation(csv_path: str | Path) -> pd.DataFrame:
    """Load an EMS annotation CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def _iter_strains(df: pd.DataFrame, column: str = STRAIN_COLUMN):
    """Yield individual strain IDs from a whitespace-separated STRAIN column."""
    for cell in df[column].dropna():
        for strain in str(cell).split():
            if strain:
                yield strain


def count_unique_strains(df: pd.DataFrame, column: str = STRAIN_COLUMN) -> int:
    """Return the number of unique strains across all rows.

    Each row's STRAIN field may list multiple strains separated by whitespace;
    all of them are split out before counting.
    """
    return len(set(_iter_strains(df, column)))


def list_unique_strains(df: pd.DataFrame, column: str = STRAIN_COLUMN) -> list[str]:
    """Return a sorted list of unique strain identifiers across all rows."""
    return sorted(set(_iter_strains(df, column)))


def count_unique_consequences(
    df: pd.DataFrame, column: str = CONSEQUENCE_COLUMN
) -> int:
    """Return the number of unique CONSEQUENCE categories."""
    return df[column].nunique(dropna=True)


def consequence_counts(
    df: pd.DataFrame, column: str = CONSEQUENCE_COLUMN
) -> pd.Series:
    """Return a Series of variant counts per CONSEQUENCE category (descending)."""
    return df[column].value_counts(dropna=False)


def count_unique_impacts(df: pd.DataFrame, column: str = IMPACT_COLUMN) -> int:
    """Return the number of unique IMPACT categories (excluding N/A)."""
    return df[column].replace("N/A", pd.NA).nunique(dropna=True)


def impact_counts(df: pd.DataFrame, column: str = IMPACT_COLUMN) -> pd.Series:
    """Return a Series of variant counts per IMPACT category (descending)."""
    return df[column].value_counts(dropna=False)


def consequence_impact_matrix(
    df: pd.DataFrame,
    consequence_column: str = CONSEQUENCE_COLUMN,
    impact_column: str = IMPACT_COLUMN,
) -> pd.DataFrame:
    """Return a CONSEQUENCE x IMPACT count matrix.

    Rows are CONSEQUENCE categories, columns are IMPACT categories, and each
    cell is the number of variants with that (CONSEQUENCE, IMPACT) pair.
    Row/column totals are appended as an ``All`` margin.
    """
    return pd.crosstab(
        df[consequence_column],
        df[impact_column].fillna("N/A"),
        margins=True,
        margins_name="All",
        dropna=False,
    )


def summarize(df: pd.DataFrame) -> dict:
    """Return a small dictionary summary of the dataset.

    Impact-related entries are only included when the ``IMPACT`` column is
    present in the DataFrame.
    """
    summary: dict = {
        "n_variants": len(df),
        "n_unique_strains": count_unique_strains(df),
        "strains": list_unique_strains(df),
        "n_unique_consequences": count_unique_consequences(df),
        "consequence_counts": consequence_counts(df),
        "has_impact": IMPACT_COLUMN in df.columns,
    }
    if summary["has_impact"]:
        summary["n_unique_impacts"] = count_unique_impacts(df)
        summary["impact_counts"] = impact_counts(df)
        summary["consequence_impact_matrix"] = consequence_impact_matrix(df)
    return summary


def main() -> None:
    df = load_annotation(CSV_PATH)
    summary = summarize(df)

    print(f"File:              {CSV_PATH}")
    print(f"Total variants:    {summary['n_variants']}")
    print(f"Unique strains:    {summary['n_unique_strains']}")
    print("Strains:")
    for s in summary["strains"]:
        print(f"  - {s}")

    print(f"Unique consequences: {summary['n_unique_consequences']}")
    print("Consequence counts:")
    for cons, n in summary["consequence_counts"].items():
        print(f"  - {cons}: {n}")

    if summary["has_impact"]:
        print(f"Unique impacts (excluding N/A): {summary['n_unique_impacts']}")
        print("Impact counts:")
        for imp, n in summary["impact_counts"].items():
            print(f"  - {imp}: {n}")

        print("CONSEQUENCE x IMPACT matrix:")
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 200,
        ):
            print(summary["consequence_impact_matrix"])
    else:
        print(
            f"(No '{IMPACT_COLUMN}' column found — skipping impact summaries.)"
        )


if __name__ == "__main__":
    main()