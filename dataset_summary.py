"""Dataset summary utilities for EMS annotation CSV files.

Edit the parameters in the CONFIG section below and run:

    python dataset_summary.py

Additional summary functions can be added to this module over time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG — edit these parameters as needed
# --------------------------------------------------------------------------- #
CSV_PATH = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\EMS_annotation\datasets\c_elegans"
    r"\c_elegans.annovar.EMS_annotation.csv"
)
STRAIN_COLUMN = "STRAIN"
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


def summarize(df: pd.DataFrame) -> dict:
    """Return a small dictionary summary of the dataset."""
    return {
        "n_variants": len(df),
        "n_unique_strains": count_unique_strains(df),
        "strains": list_unique_strains(df),
    }


def main() -> None:
    df = load_annotation(CSV_PATH)
    summary = summarize(df)

    print(f"File:              {CSV_PATH}")
    print(f"Total variants:    {summary['n_variants']}")
    print(f"Unique strains:    {summary['n_unique_strains']}")
    print("Strains:")
    for s in summary["strains"]:
        print(f"  - {s}")


if __name__ == "__main__":
    main()
