"""Matplotlib visualisations — ports of tigger's ``plotNovel`` and
``plotGenotype``.

* :func:`plot_novel`    — three-panel evidence plot for a novel allele:
  position mutation frequency vs sequence mutation count, nucleotide usage
  at polymorphic positions, and J-gene / junction-length diversity.
* :func:`plot_genotype` — coloured genotype grid (one bar per gene).
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from .novel import _mutation_range_subset, _position_mutations
from .segments import get_gene
from .sequences import clean_seqs, get_mutated_positions, sort_alleles

__all__ = ["plot_novel", "plot_genotype"]

# alakazam DNA_COLORS.
DNA_COLORS = {"A": "#64F73F", "C": "#FFB340", "G": "#EB413C", "T": "#3C88EE"}


def plot_novel(data: pd.DataFrame, novel_row, v_call: str = "v_call",
               j_call: str = "j_call", seq: str = "sequence_alignment",
               junction: str = "junction",
               junction_length: str = "junction_length",
               pos_range_max: Optional[str] = None, ncol: int = 1,
               multiplot: bool = True):
    """Visualise the evidence supporting a novel V allele.

    Port of ``tigger::plotNovel``.  ``novel_row`` is a single-row
    :class:`pandas.DataFrame` from :func:`find_novel_alleles`.  Returns a
    matplotlib ``Figure`` when ``multiplot`` is ``True`` else a list of
    three ``Figure`` objects.
    """
    import matplotlib.pyplot as plt

    if not isinstance(novel_row, pd.DataFrame) or len(novel_row) != 1:
        raise ValueError("novel_row is not a data frame with only one row.")
    row = novel_row.iloc[0]
    pos_range = list(range(int(row["pos_min"]), int(row["pos_max"]) + 1))
    germline = clean_seqs(str(row["germline_imgt"]))
    germ_name = str(row["germline_call"])
    mut_range = list(range(int(row["mut_min"]), int(row["mut_max"]) + 1))
    novel_imgt = row["novel_imgt"]
    poly_call = row["polymorphism_call"]
    min_frac = float(row["min_frac"])
    note = str(row["note"]) if not pd.isna(row["note"]) else ""

    df = data.copy()
    df[seq] = clean_seqs(list(df[seq]))
    db_subset = df[df[v_call].astype(str).str.contains(germ_name, regex=False)]
    db_subset = db_subset.reset_index(drop=True).copy()

    pos_db = _mutation_range_subset(db_subset, germline, mut_range, pos_range,
                                    seq=seq, pos_range_max=pos_range_max)
    if len(pos_db) == 0:
        import warnings
        warnings.warn("Insufficient sequences in desired mutational range.")
        return None
    pos_db = _position_mutations(pos_db, germline, pos_range, seq=seq,
                                 pos_range_max=pos_range_max)

    pass_by_pos = (pos_db.groupby("POSITION")["OBSERVED"].mean() >= min_frac)
    pos_db = pos_db.copy()
    pos_db["PASS"] = pos_db["POSITION"].map(pass_by_pos).astype(float)
    grp = pos_db.groupby(["MUT_COUNT", "POSITION"], sort=True)
    pos_muts = grp.apply(lambda g: pd.Series({
        "POS_MUT_RATE": g["MUTATED"].mean() * g["PASS"].iloc[0]
    }), include_groups=False).reset_index()

    # Polymorphic positions from the polymorphism_call name.
    pass_y = []
    if not pd.isna(poly_call):
        parts = str(poly_call).split("_")[1:]
        for token in parts:
            num = re.sub(r"[^0-9]", "", token)
            if num:
                pass_y.append(int(num))
    if not pass_y and "Position(s) passed y-intercept" in note:
        inside = re.sub(r"Position\(s\) passed y-intercept \(", "", note)
        inside = re.sub(r"\).*", "", inside)
        pass_y = [int(x) for x in inside.split(",") if x.strip()]

    pos_muts["Polymorphic"] = np.where(
        pos_muts["POSITION"].isin(pass_y), "True", "False")

    # MUT_COUNT_NOVEL for the third panel.
    pads = "-" * (min(pos_range) - 1)
    if not pd.isna(novel_imgt):
        subseq = db_subset[seq].astype(str).str.slice(
            min(pos_range) - 1, max(pos_range))
        padded = [pads + s for s in subseq]
        mc_novel = [len(p) for p in get_mutated_positions(
            padded, str(novel_imgt))]
        db_subset = db_subset.copy()
        db_subset["MUT_COUNT_NOVEL"] = mc_novel
        db3 = db_subset[db_subset["MUT_COUNT_NOVEL"] == 0].copy()
    else:
        db3 = db_subset.copy()
    if len(db3) > 0:
        db3["J_GENE"] = get_gene(list(db3[j_call]))

    x_min, x_max = pos_muts["MUT_COUNT"].min(), pos_muts["MUT_COUNT"].max()

    # ---- Panel 1: position mutation frequency vs sequence mutation count.
    def _make_p1(ax):
        novel_found = not pd.isna(novel_imgt)
        true_color = DNA_COLORS["G"] if novel_found else DNA_COLORS["C"]
        false_color = DNA_COLORS["T"]
        for poly_state, color, lw in (("False", false_color, 0.75),
                                      ("True", true_color, 0.75)):
            sub = pos_muts[pos_muts["Polymorphic"] == poly_state]
            for pos, g in sub.groupby("POSITION"):
                g = g.sort_values("MUT_COUNT")
                ax.plot(g["MUT_COUNT"], g["POS_MUT_RATE"], color=color,
                        linewidth=lw)
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mutation Count (Sequence)")
        ax.set_ylabel("Mutation Frequency (Position)")
        ax.set_title(germ_name)
        import matplotlib.lines as mlines
        handles = [
            mlines.Line2D([], [], color=true_color, label="True"),
            mlines.Line2D([], [], color=false_color, label="False"),
        ]
        legtitle = (None if novel_found else "Passed y-intercept test")
        ax.legend(handles=handles, title=legtitle, loc="upper center",
                  fontsize=8, ncol=2)

    # ---- Panel 2: nucleotide usage at polymorphic positions.
    def _make_p2(ax):
        p2_data = pos_db[pos_db["POSITION"].isin(pass_y)]
        if len(p2_data) > 0:
            positions = sorted(p2_data["POSITION"].unique())
            width = 0.8 / max(1, len(positions))
            mut_counts = sorted(p2_data["MUT_COUNT"].unique())
            # Stacked bar per nucleotide.
            for pi, pos in enumerate(positions):
                pd_sub = p2_data[p2_data["POSITION"] == pos]
                bottom = np.zeros(len(mut_counts))
                for nt in ["A", "C", "G", "T"]:
                    counts = []
                    for mc in mut_counts:
                        counts.append(int(((pd_sub["MUT_COUNT"] == mc)
                                           & (pd_sub["NT"] == nt)).sum()))
                    xs = np.array(mut_counts) + pi * width
                    ax.bar(xs, counts, width=width, bottom=bottom,
                           color=DNA_COLORS[nt],
                           label=nt if pi == 0 else None)
                    bottom += np.array(counts)
            ax.legend(title="Nucleotide", fontsize=8, ncol=4)
        else:
            # No polymorphisms — plot mutation count distribution.
            top_pos = pos_db["POSITION"].value_counts().idxmax()
            pd_sub = pos_db[pos_db["POSITION"] == top_pos]
            mut_counts = sorted(pd_sub["MUT_COUNT"].unique())
            counts = [int((pd_sub["MUT_COUNT"] == mc).sum())
                      for mc in mut_counts]
            ax.bar(mut_counts, counts, width=0.9, color="grey")
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_xlabel("Mutation Count (Sequence)")
        ax.set_ylabel("Sequence Count")

    # ---- Panel 3: J-gene / junction-length diversity.
    def _make_p3(ax):
        if len(db3) == 0:
            ax.text(0.5, 0.5, "No unmutated sequences",
                    ha="center", va="center")
            return
        jl = db3[junction_length].astype(int)
        jl_min, jl_max = jl.min(), jl.max()
        lengths = list(range(jl_min, jl_max + 1))
        j_genes = sorted(db3["J_GENE"].unique())
        bottom = np.zeros(len(lengths))
        cmap = __import__("matplotlib.cm", fromlist=["get_cmap"])
        colors = cmap.get_cmap("tab20")(np.linspace(0, 1, len(j_genes)))
        for ji, jg in enumerate(j_genes):
            counts = []
            for L in lengths:
                counts.append(int(((db3["J_GENE"] == jg)
                                   & (jl == L)).sum()))
            ax.bar(lengths, counts, width=0.9, bottom=bottom,
                   color=colors[ji], label=jg)
            bottom += np.array(counts)
        ax.set_xlabel("Junction Length")
        ax.set_ylabel("Unmutated Sequence Count")
        ax.legend(title="J Gene", fontsize=7, ncol=2)

    if multiplot:
        fig, axes = plt.subplots(3, ncol, figsize=(7, 11))
        axes = np.atleast_1d(axes).ravel()
        _make_p1(axes[0])
        _make_p2(axes[1])
        _make_p3(axes[2])
        fig.tight_layout()
        return fig
    figs = []
    for maker in (_make_p1, _make_p2, _make_p3):
        f, ax = plt.subplots(figsize=(6, 4))
        maker(ax)
        f.tight_layout()
        figs.append(f)
    return figs


def plot_genotype(genotype: pd.DataFrame, facet_by: Optional[str] = None,
                  gene_sort: str = "name", text_size: int = 12,
                  silent: bool = False):
    """Plot a coloured genotype grid.

    Port of ``tigger::plotGenotype``.  Each gene is a horizontal bar split
    into equal-width segments coloured by allele.  Returns the matplotlib
    ``Figure``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if gene_sort.startswith("pos"):
        gene_sort = "position"
    elif gene_sort.startswith("nam"):
        gene_sort = "name"

    # Split alleles into their own rows.
    rows = []
    for _, r in genotype.iterrows():
        for a in str(r["alleles"]).split(","):
            new = dict(r)
            new["alleles"] = a
            rows.append(new)
    geno2 = pd.DataFrame(rows)

    # Gene order (reversed sortAlleles so first gene is at top).
    gene_order = sort_alleles(list(pd.unique(geno2["gene"])),
                              method=gene_sort)
    gene_order = list(reversed(gene_order))

    all_alleles = sorted(pd.unique(geno2["alleles"]))
    cmap = plt.get_cmap("hsv")
    allele_colors = {
        a: cmap((i / max(1, len(all_alleles))) * 0.75)
        for i, a in enumerate(all_alleles)
    }

    facets = ([None] if facet_by is None
              else list(pd.unique(genotype[facet_by])))
    fig, axes = plt.subplots(1, len(facets),
                             figsize=(3 * len(facets) + 2,
                                      0.35 * len(gene_order) + 1),
                             squeeze=False)
    for fi, facet in enumerate(facets):
        ax = axes[0][fi]
        sub = geno2 if facet is None else geno2[geno2[facet_by] == facet]
        for yi, gene in enumerate(gene_order):
            galleles = list(sub[sub["gene"] == gene]["alleles"])
            if not galleles:
                continue
            seg = 1.0 / len(galleles)
            for si, a in enumerate(galleles):
                ax.barh(yi, seg, left=si * seg, height=0.8,
                        color=allele_colors[a], edgecolor="white")
        ax.set_yticks(range(len(gene_order)))
        ax.set_yticklabels(gene_order, fontsize=text_size - 2)
        ax.set_xticks([])
        ax.set_xlim(0, 1)
        ax.set_ylabel("Gene", fontsize=text_size)
        if facet is not None:
            ax.set_title(str(facet), fontsize=text_size, fontweight="bold")
    handles = [Patch(color=allele_colors[a], label=a) for a in all_alleles]
    fig.legend(handles=handles, title="Allele", loc="center right",
               fontsize=text_size - 3)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    if not silent:
        plt.draw()
    return fig
