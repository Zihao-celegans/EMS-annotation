"""Plot number of strains carrying EMS-induced variants per gene.

For each gene (WBGENE), count the number of distinct strains carrying at least
one EMS-signature variant (``possible_EMS == "yes"``) whose CONSEQUENCE is not
in the excluded set ``{intergenic, upstream, downstream}``. Genes are ordered
along the x-axis by genome position (chromosome, then minimum POS in the gene);
the y-axis is the per-gene unique-strain count.

Edit the parameters in the CONFIG section below and run:

    python plot_gene_strain_counts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG — edit these parameters as needed
# --------------------------------------------------------------------------- #
CSV_PATH = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\EMS_annotation\datasets\c_elegans"
    r"\c_elegans.annovar.EMS_annotation.csv"
)
OUTPUT_DIR = Path(r"C:\Users\jl200\Dropbox\JHU_2026_spring\EMS_annotation\analysis")
TOP_N = 10

# Strain groups by drug screen. Per-screen results are written separately.
STRAIN_GROUPS: dict[str, list[str]] = {
    "Kelleher": [
        "ECA4365", "ECA4366", "ECA4367", "ECA4368", "ECA4369",
        "ECA4370", "ECA4371", "ECA4372", "ECA4373",
    ],
    "ABZ": [
        "ECA4236", "ECA4237", "ECA4238", "ECA4239", "ECA4240", "ECA4241",
        "ECA4242", "ECA4243", "ECA4244", "ECA4245", "ECA4246",
    ],
}

EXCLUDED_CONSEQUENCES = {"intergenic", "upstream", "downstream"}

# C. elegans chromosome ordering for x-axis placement
CHROM_ORDER = ["I", "II", "III", "IV", "V", "X", "MtDNA"]

STRAIN_COLUMN = "STRAIN"
GENE_COLUMN = "WBGENE"
GENE_NAME_COLUMN = "GENE_NAME"
CONSEQUENCE_COLUMN = "CONSEQUENCE"
CHROM_COLUMN = "CHROM"
POS_COLUMN = "POS"
POSSIBLE_EMS_COLUMN = "possible_EMS"
# --------------------------------------------------------------------------- #


def load_annotation(csv_path: str | Path) -> pd.DataFrame:
    """Load an EMS annotation CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def infer_species(csv_path: str | Path) -> str:
    """Infer the species tag from the CSV filename (text before the first '.').

    Example: ``c_elegans.annovar.EMS_annotation.csv`` -> ``c_elegans``.
    """
    return Path(csv_path).name.split(".")[0]


def filter_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Keep EMS-signature variants with a non-excluded CONSEQUENCE."""
    is_ems = df[POSSIBLE_EMS_COLUMN].astype(str).str.lower() == "yes"
    keep_cons = ~df[CONSEQUENCE_COLUMN].isin(EXCLUDED_CONSEQUENCES)
    has_gene = df[GENE_COLUMN].notna() & (df[GENE_COLUMN].astype(str) != "N/A")
    return df.loc[is_ems & keep_cons & has_gene].copy()


def _split_strains(cell: object) -> list[str]:
    if pd.isna(cell):
        return []
    return [s for s in str(cell).split() if s]


def gene_strain_counts(
    df: pd.DataFrame,
    allowed_strains: set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-gene unique strain counts and genome position.

    If ``allowed_strains`` is provided, only those strain IDs are counted, and
    genes whose qualifying variants belong exclusively to other strains are
    dropped.

    Returns a DataFrame with columns:
        WBGENE, GENE_NAME, CHROM, min_POS, n_strains
    sorted by (CHROM order, min_POS).
    """
    records = []
    for wbgene, sub in df.groupby(GENE_COLUMN, dropna=True):
        strains: set[str] = set()
        for cell in sub[STRAIN_COLUMN]:
            for s in _split_strains(cell):
                if allowed_strains is None or s in allowed_strains:
                    strains.add(s)

        if not strains:
            continue

        # Pick a representative common name (first non-missing, non-"N/A")
        name_candidates = sub[GENE_NAME_COLUMN].dropna()
        name_candidates = name_candidates[
            ~name_candidates.astype(str).isin({"N/A", "nan", ""})
        ]
        gene_name = name_candidates.iloc[0] if not name_candidates.empty else "N/A"

        records.append(
            {
                GENE_COLUMN: wbgene,
                GENE_NAME_COLUMN: gene_name,
                CHROM_COLUMN: sub[CHROM_COLUMN].iloc[0],
                "min_POS": int(sub[POS_COLUMN].min()),
                "n_strains": len(strains),
            }
        )

    out = pd.DataFrame.from_records(records)

    # Order chromosomes using CHROM_ORDER; unknown chroms go to the end.
    chrom_rank = {c: i for i, c in enumerate(CHROM_ORDER)}
    out["_chrom_rank"] = out[CHROM_COLUMN].map(
        lambda c: chrom_rank.get(c, len(CHROM_ORDER))
    )
    out = out.sort_values(["_chrom_rank", "min_POS"]).drop(columns="_chrom_rank")
    out = out.reset_index(drop=True)
    return out


def plot_counts(gene_df: pd.DataFrame, output_png: Path, title_suffix: str = "") -> None:
    """Scatter plot of per-gene strain counts along genome-position-ordered index."""
    x = range(1, len(gene_df) + 1)
    y = gene_df["n_strains"].to_numpy()

    # Color by chromosome for quick visual separation.
    chroms_present = [c for c in CHROM_ORDER if c in set(gene_df[CHROM_COLUMN])]
    cmap = plt.get_cmap("tab10")
    color_map = {c: cmap(i % 10) for i, c in enumerate(chroms_present)}
    colors = gene_df[CHROM_COLUMN].map(color_map).tolist()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(list(x), y, c=colors, s=10, alpha=0.7)

    ax.set_xlabel("Gene index (ordered by chromosome, then genome position)")
    ax.set_ylabel("Number of strains carrying ≥1 qualifying EMS variant")
    title = (
        f"Per-gene strain counts "
        f"(EMS-signature, excluding {sorted(EXCLUDED_CONSEQUENCES)})"
    )
    if title_suffix:
        title = f"{title_suffix} — " + title
    ax.set_title(title)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=color_map[c], label=c, markersize=6
        )
        for c in chroms_present
    ]
    ax.legend(handles=handles, title="CHROM", loc="best", fontsize=8)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    print(f"Saved figure to: {output_png}")


def run_for_group(
    filtered: pd.DataFrame,
    group_name: str,
    group_strains: list[str],
    species: str,
) -> None:
    print(f"\n=== {species} | {group_name} ({len(group_strains)} strains) ===")
    allowed = set(group_strains)
    gene_df = gene_strain_counts(filtered, allowed_strains=allowed)
    print(f"Genes with ≥1 qualifying variant in this group: {len(gene_df)}")

    top_genes = gene_df.sort_values("n_strains", ascending=False).head(TOP_N)
    print(f"Top {TOP_N} genes by strain count:")
    print(top_genes.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top_csv = (
        OUTPUT_DIR
        / f"{species}_top{TOP_N}_genes_by_strain_count_{group_name}.csv"
    )
    top_genes.to_csv(top_csv, index=False)
    print(f"Saved top-{TOP_N} table to: {top_csv}")

    png = OUTPUT_DIR / f"{species}_gene_strain_counts_{group_name}.png"
    plot_counts(gene_df, png, title_suffix=f"{species} | {group_name}")


def main() -> None:
    df = load_annotation(CSV_PATH)
    filtered = filter_variants(df)
    species = infer_species(CSV_PATH)
    print(f"Species:               {species}")
    print(f"Total rows:            {len(df)}")
    print(f"After filtering:       {len(filtered)}")

    for group_name, group_strains in STRAIN_GROUPS.items():
        run_for_group(filtered, group_name, group_strains, species)

    plt.show()


if __name__ == "__main__":
    main()
