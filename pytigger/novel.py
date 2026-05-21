"""Novel V-allele discovery — faithful port of tigger's ``findNovelAlleles``.

The TIgGER allele-finding algorithm analyses mutation patterns in sequences
assigned to each germline allele, identifies positions that are polymorphic
(high mutation frequency despite low sequence-wide mutation count) via a
y-intercept regression, and confirms candidates with a J-gene / junction
diversity test.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .segments import get_gene
from .sequences import (
    clean_seqs,
    get_mut_count,
    get_mutated_positions,
    insert_polymorphisms,
    sort_alleles,
    super_substring,
)

__all__ = [
    "find_novel_alleles",
    "select_novel",
    "get_popular_mutation_count",
]

# Column order of the findNovelAlleles output (matches tigger).
_NOVEL_COLUMNS = [
    "germline_call", "note", "polymorphism_call", "nt_substitutions",
    "novel_imgt", "novel_imgt_count", "novel_imgt_unique_j",
    "novel_imgt_unique_cdr3", "perfect_match_count", "perfect_match_freq",
    "germline_call_count", "germline_call_freq", "mut_min", "mut_max",
    "mut_pass_count", "germline_imgt", "germline_imgt_count", "pos_min",
    "pos_max", "y_intercept", "y_intercept_pass", "snp_pass",
    "unmutated_count", "unmutated_freq", "unmutated_snp_j_gene_length_count",
    "snp_min_seqs_j_max_pass", "alpha", "min_seqs", "j_max", "min_frac",
]


# ---------------------------------------------------------------------------
def get_popular_mutation_count(data: pd.DataFrame, germline_db: Dict[str, str],
                               v_call: str = "v_call",
                               seq: str = "sequence_alignment",
                               gene_min: float = 1e-3,
                               seq_min: int = 50,
                               seq_p_of_max: float = 1 / 8,
                               full_return: bool = False) -> pd.DataFrame:
    """Find mutation counts of frequently-occurring V sequences.

    Port of ``tigger::getPopularMutationCount``.
    """
    n_total = len(data)
    df = data.copy()
    df["v_gene"] = get_gene(list(df[v_call]))
    # v_gene_n: count per gene
    df["v_gene_n"] = df.groupby("v_gene")["v_gene"].transform("size")
    # IMGT-gapped V subsequence (first 312 nt)
    df["v_sequence_imgt"] = df[seq].astype(str).str.slice(0, 312)
    # count of each unique (gene, sequence)
    df["v_sequence_imgt_n"] = df.groupby(
        ["v_gene", "v_sequence_imgt"])["v_sequence_imgt"].transform("size")
    # max sequence count per gene
    df["v_sequence_imgt_n_max"] = df.groupby(
        "v_gene")["v_sequence_imgt_n"].transform("max")
    # distinct(v_sequence_imgt, .keep_all=TRUE) — keep first row per sequence
    df = df.drop_duplicates(subset="v_sequence_imgt", keep="first")
    df = df[df["v_gene_n"] >= n_total * gene_min]
    df = df[df["v_sequence_imgt_n"] >= seq_min]
    df = df.copy()
    df["v_sequence_imgt_p_max"] = (
        df["v_sequence_imgt_n"] / df["v_sequence_imgt_n_max"])
    df = df[df["v_sequence_imgt_p_max"] >= seq_p_of_max]

    if len(df) == 0:
        mutation_count = []
    else:
        mc = get_mut_count(list(df["v_sequence_imgt"]),
                           list(df[v_call]), germline_db)
        mutation_count = []
        for x in mc:
            if isinstance(x, dict):
                mutation_count.append(min(x.values()))
            else:
                mutation_count.append(x)
    df = df.copy()
    df["mutation_count"] = mutation_count

    if not full_return:
        df = df[df["mutation_count"] > 0][["v_gene", "mutation_count"]]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
def _mutation_range_subset(data: pd.DataFrame, germline: str, mut_range,
                           pos_range, seq: str = "sequence_alignment",
                           pos_range_max: Optional[str] = None) -> pd.DataFrame:
    """Subset sequences whose mutation count falls in ``mut_range``.

    Port of tigger's private ``mutationRangeSubset``.
    """
    pos_min = min(pos_range)
    pos_max = max(pos_range)
    pads = "-" * (pos_min - 1)
    seqs = data[seq].astype(str)
    if pos_range_max is None:
        subseq = seqs.str.slice(pos_min - 1, pos_max)
    else:
        subseq = [
            s[pos_min - 1:min(pos_max, int(pm))]
            for s, pm in zip(seqs, data[pos_range_max])
        ]
    padded = [pads + s for s in subseq]
    mut_count = [len(p) for p in get_mutated_positions(padded, germline)]
    out = data.copy()
    out["MUT_COUNT"] = mut_count
    mr = set(mut_range)
    return out[out["MUT_COUNT"].isin(mr)].copy()


def _position_mutations(data: pd.DataFrame, germline: str, pos_range,
                        seq: str = "sequence_alignment",
                        pos_range_max: Optional[str] = None) -> pd.DataFrame:
    """Long-format table: one row per (sequence, position) with mutation flag.

    Port of tigger's private ``positionMutations``.
    """
    germ = str(germline)
    seqs = data[seq].astype(str).tolist()
    mut_count = data["MUT_COUNT"].tolist()
    pmax = data[pos_range_max].tolist() if pos_range_max is not None else None

    rows_seq = []
    rows_pos = []
    rows_nt = []
    rows_germnt = []
    rows_mc = []
    rows_pm = []
    for pos in pos_range:
        gnt = germ[pos - 1] if 0 <= pos - 1 < len(germ) else ""
        for i, s in enumerate(seqs):
            nt = s[pos - 1] if 0 <= pos - 1 < len(s) else ""
            rows_seq.append(i)
            rows_pos.append(pos)
            rows_nt.append(nt)
            rows_germnt.append(gnt)
            rows_mc.append(mut_count[i])
            if pmax is not None:
                rows_pm.append(pmax[i])

    df = pd.DataFrame({
        "SEQ_IDX": rows_seq,
        "POSITION": rows_pos,
        "NT": rows_nt,
        "GERM_NT": rows_germnt,
        "MUT_COUNT": rows_mc,
    })
    df["MUTATED"] = ((df["NT"] != df["GERM_NT"]) & (df["NT"] != "N") &
                     (df["NT"] != "-") & (df["NT"] != ".") & (df["NT"] != ""))
    df["OBSERVED"] = ((df["NT"] != "-") & (df["NT"] != ".") & (df["NT"] != ""))
    if pos_range_max is not None:
        df["_PMAX"] = rows_pm
        df = df[df["POSITION"] <= df["_PMAX"]].drop(columns="_PMAX")
    if (df["GERM_NT"] == "").any() and "." not in germ:
        raise ValueError(
            "Empty ('') GERM_NT positions found. Check you are using "
            "gapped reference germlines.")
    return df


def _find_lower_y(x, y, mut_min: int, alpha: float) -> float:
    """Lower bound of the y-intercept confidence interval.

    Port of tigger's private ``findLowerY``.  Fits ``lm(x ~ y)`` where
    ``y`` has been shifted so the intercept is evaluated at ``mut_min``,
    and returns the lower edge of the ``1 - 2*alpha`` confidence interval
    for the intercept.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float) + 1.0 - mut_min
    n = len(x)
    if n < 2:
        # lm with <2 points: R returns NA -> propagate as -inf so it fails
        return float("-inf")
    # OLS: x = b0 + b1 * y
    ybar = y.mean()
    xbar = x.mean()
    sxy = np.sum((y - ybar) * (x - xbar))
    syy = np.sum((y - ybar) ** 2)
    if syy == 0:
        return float("-inf")
    b1 = sxy / syy
    b0 = xbar - b1 * ybar
    resid = x - (b0 + b1 * y)
    dof = n - 2
    if dof <= 0:
        return float("-inf")
    sigma2 = np.sum(resid ** 2) / dof
    # se of intercept
    se_b0 = np.sqrt(sigma2 * (1.0 / n + ybar ** 2 / syy))
    from scipy.stats import t as _t
    # confint level = 1 - 2*alpha -> two-sided; lower quantile is alpha
    tcrit = _t.ppf(1.0 - alpha, dof)
    return b0 - tcrit * se_b0


# ---------------------------------------------------------------------------
def _mu_spec(poly_call: str, germ_call: str) -> str:
    """Format nucleotide-substitution string, e.g. ``234G>T``."""
    p = poly_call.replace(germ_call, "")
    parts = p.split("_")[1:]
    out = []
    for token in parts:
        m = re.match(r"([A-Za-z])([0-9]*)([A-Za-z])", token)
        if m:
            out.append(f"{m.group(2)}{m.group(1)}>{m.group(3)}")
        else:
            out.append(token)
    return ",".join(out)


def find_novel_alleles(data: pd.DataFrame, germline_db: Dict[str, str],
                       v_call: str = "v_call", j_call: str = "j_call",
                       seq: str = "sequence_alignment",
                       junction: str = "junction",
                       junction_length: str = "junction_length",
                       germline_min: int = 200, min_seqs: int = 50,
                       auto_mutrange: bool = True,
                       mut_range=range(1, 11),
                       pos_range=range(1, 313),
                       pos_range_max: Optional[str] = None,
                       y_intercept: float = 0.125, alpha: float = 0.05,
                       j_max: float = 0.15, min_frac: float = 0.75,
                       nproc: int = 1) -> pd.DataFrame:
    """Find novel V alleles from repertoire sequencing data.

    Faithful port of ``tigger::findNovelAlleles``.  Returns a
    :class:`pandas.DataFrame` with one row per known allele analysed,
    matching tigger's output columns.
    """
    mut_range = list(mut_range)
    pos_range = list(pos_range)
    pos_min_r, pos_max_r = min(pos_range), max(pos_range)

    # Keep only needed columns; check presence.
    needed = [seq, v_call, j_call, junction_length]
    if pos_range_max is not None:
        needed.append(pos_range_max)
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(
            "Could not find required columns in the input data:\n  "
            + "\n  ".join(missing))

    cols = [seq, v_call, j_call, junction_length, junction]
    if pos_range_max is not None:
        cols.append(pos_range_max)
    cols = [c for c in cols if c in data.columns]
    data = data[cols].reset_index(drop=True).copy()

    empty_junctions = int((data[junction_length] == 0).sum())
    if empty_junctions > 0:
        raise ValueError(
            f"{empty_junctions} sequences have junction length of zero. "
            "Please remove these sequences.")

    germlines = clean_seqs(dict(germline_db))
    # Rename keys via getAllele (first=False, strip_d=False).
    from .segments import get_allele
    new_germlines = {}
    for k, v in germlines.items():
        new_germlines[get_allele(k, first=False, strip_d=False)] = v
    germlines = new_germlines

    data[seq] = clean_seqs(list(data[seq]))

    # Cutoff for germline_min.
    cutoff = (round(len(data) * germline_min)
              if germline_min < 1 else germline_min)

    v_calls = list(data[v_call])
    allele_groups: Dict[str, List[int]] = {}
    for name in germlines:
        idx = [i for i, vc in enumerate(v_calls) if name in str(vc)]
        allele_groups[name] = idx
    allele_groups = {k: v for k, v in allele_groups.items()
                     if len(v) >= cutoff}
    if not allele_groups:
        raise ValueError(
            "Not enough sample sequences were assigned to any germline:\n"
            "  (1) germline_min is too large or\n"
            "  (2) sequences names don't match germlines.")
    ordered_names = sort_alleles(list(allele_groups.keys()))
    allele_groups = {k: allele_groups[k] for k in ordered_names}

    out_frames = []
    for allele_name in allele_groups:
        germ = germlines[allele_name]
        indices = allele_groups[allele_name]
        db_subset = data.iloc[indices].reset_index(drop=True).copy()

        # Popular mutation count for auto mutrange.
        gpm_in = db_subset.copy()
        gpm_in[v_call] = allele_name
        gpm = get_popular_mutation_count(
            gpm_in, {allele_name: germ}, gene_min=0, seq_min=min_seqs,
            seq_p_of_max=1 / 8, full_return=True, v_call=v_call, seq=seq)

        mut_mins = [min(mut_range)]
        if auto_mutrange and (gpm["mutation_count"] > 0).sum() > 0:
            extra = sorted(set(
                gpm["mutation_count"][gpm["mutation_count"] > 0].tolist()))
            mut_mins = sorted(set(mut_mins + extra))

        df_run_empty = {
            "germline_call": allele_name, "note": "",
            "polymorphism_call": np.nan, "nt_substitutions": np.nan,
            "novel_imgt": np.nan, "novel_imgt_count": np.nan,
            "novel_imgt_unique_j": np.nan, "novel_imgt_unique_cdr3": np.nan,
            "perfect_match_count": np.nan, "perfect_match_freq": np.nan,
            "germline_call_count": len(indices),
            "germline_call_freq": round(len(indices) / len(data), 3),
            "mut_min": np.nan, "mut_max": np.nan, "mut_pass_count": np.nan,
            "germline_imgt": str(germ), "germline_imgt_count": np.nan,
            "pos_min": pos_min_r, "pos_max": pos_max_r,
            "y_intercept": y_intercept, "y_intercept_pass": np.nan,
            "snp_pass": np.nan, "unmutated_count": np.nan,
            "unmutated_freq": np.nan,
            "unmutated_snp_j_gene_length_count": np.nan,
            "snp_min_seqs_j_max_pass": np.nan, "alpha": alpha,
            "min_seqs": min_seqs, "j_max": j_max, "min_frac": min_frac,
        }

        # df_run is a list of dict rows; row 0 is the "current" one.
        df_run: List[dict] = []
        rev_mins = list(reversed(mut_mins))
        for mi, mut_min in enumerate(rev_mins):
            if mi == 0:
                df_run = [dict(df_run_empty)]
            else:
                df_run = [dict(df_run_empty)] + df_run
            mut_max = mut_min + (max(mut_range) - min(mut_range))
            df_run[0]["mut_min"] = mut_min
            df_run[0]["mut_max"] = mut_max

            last_iter = (mut_mins[0] == mut_min)

            if len(gpm) < 1:
                df_run[0]["note"] = "Plurality sequence too rare."
                if last_iter:
                    break
                continue

            db_subset_mm = _mutation_range_subset(
                db_subset, germ, range(mut_min, mut_max + 1), pos_range,
                seq=seq, pos_range_max=pos_range_max)
            df_run[0]["mut_pass_count"] = len(db_subset_mm)

            if len(db_subset_mm) < min_seqs:
                df_run[0]["note"] = (
                    f"Insufficient sequences ({len(db_subset_mm)}) in "
                    "desired mutational range.")
                if last_iter:
                    break
                continue

            pos_db = _position_mutations(
                db_subset_mm, germ, pos_range, seq=seq,
                pos_range_max=pos_range_max)

            # POS_MUT_RATE per (MUT_COUNT, POSITION).
            pass_by_pos = (pos_db.groupby("POSITION")["OBSERVED"].mean()
                           >= min_frac)
            pos_db = pos_db.copy()
            pos_db["PASS"] = pos_db["POSITION"].map(pass_by_pos).astype(float)
            grp = pos_db.groupby(["MUT_COUNT", "POSITION"], sort=True)
            pos_muts = grp.apply(
                lambda g: pd.Series({
                    "POS_MUT_RATE": g["MUTATED"].mean() * g["PASS"].iloc[0]
                }), include_groups=False).reset_index()

            # y-intercept test per position.
            pass_y_rows = []
            for pos, g in pos_muts.groupby("POSITION", sort=True):
                ymin = _find_lower_y(
                    g["POS_MUT_RATE"].values, g["MUT_COUNT"].values,
                    mut_min, alpha)
                if ymin > y_intercept:
                    pass_y_rows.append(pos)
            pass_y = sorted(pass_y_rows)
            df_run[0]["y_intercept_pass"] = len(pass_y)

            if len(pass_y) < 1:
                df_run[0]["note"] = "No positions pass y-intercept test."
                if last_iter:
                    break
                continue

            gl_substring = super_substring(germ, pass_y)
            gl_minus_substring = insert_polymorphisms(
                germ, pass_y, ["N"] * len(pass_y))

            # SNP strings for each sequence in db_subset_mm.
            seqs_mm = db_subset_mm[seq].astype(str).tolist()
            snp_strings = [super_substring(s, pass_y) for s in seqs_mm]
            sub_df = db_subset_mm.copy().reset_index(drop=True)
            sub_df["SNP_STRING"] = snp_strings
            sub_df["POSITION_OK"] = True
            if pos_range_max is not None:
                max_passy = max(pass_y)
                sub_df["POSITION_OK"] = sub_df[pos_range_max] >= max_passy
            sub_df = sub_df[(sub_df["SNP_STRING"] != gl_substring)
                            & sub_df["POSITION_OK"]].copy()
            string_counts = sub_df["SNP_STRING"].value_counts()
            sub_df["STRING_COUNT"] = sub_df["SNP_STRING"].map(string_counts)
            db_y_subset_mm = sub_df[sub_df["STRING_COUNT"] >= min_seqs].copy()

            df_run[0]["snp_pass"] = len(db_y_subset_mm)

            if len(db_y_subset_mm) < 1:
                df_run[0]["note"] = (
                    "Position(s) passed y-intercept ("
                    + ",".join(str(p) for p in pass_y)
                    + ") but the plurality sequence is too rare.")
                if last_iter:
                    break
                continue

            # Mutation count outside potential SNP positions.
            pads = "-" * (pos_min_r - 1)
            seqs_y = db_y_subset_mm[seq].astype(str)
            subseq = seqs_y.str.slice(pos_min_r - 1, pos_max_r)
            padded = [pads + s for s in subseq]
            mc_minus = [len(p) for p in
                        get_mutated_positions(padded, gl_minus_substring)]
            db_y_subset_mm = db_y_subset_mm.copy()
            db_y_subset_mm["MUT_COUNT_MINUS_SUBSTRING"] = mc_minus

            db_y_summary0 = db_y_subset_mm[
                db_y_subset_mm["MUT_COUNT_MINUS_SUBSTRING"] == 0].copy()
            df_run[0]["unmutated_count"] = len(db_y_summary0)

            db_y_summary0["J_GENE"] = get_gene(list(db_y_summary0[j_call]))
            grp0 = db_y_summary0.groupby(
                ["SNP_STRING", "J_GENE", junction_length], sort=True)
            counts_df = grp0.size().reset_index(name="COUNT")
            df_run[0]["unmutated_snp_j_gene_length_count"] = len(counts_df)

            # Per SNP_STRING: TOTAL_COUNT, MAX_FRAC.
            summ_rows = []
            for snp, g in counts_df.groupby("SNP_STRING", sort=True):
                total = g["COUNT"].sum()
                max_frac = (g["COUNT"] / total).max()
                summ_rows.append({"SNP_STRING": snp, "TOTAL_COUNT": total,
                                  "MAX_FRAC": max_frac})
            db_y_summary0_df = pd.DataFrame(summ_rows)

            if len(db_y_summary0_df) < 1:
                df_run[0]["note"] = (
                    "Position(s) passed y-intercept ("
                    + ",".join(str(p) for p in pass_y)
                    + ") but no unmutated versions of novel allele found.")
                if last_iter:
                    break
                continue

            min_seqs_pass = db_y_summary0_df["TOTAL_COUNT"] >= min_seqs
            j_max_pass = db_y_summary0_df["MAX_FRAC"] <= j_max
            db_y_summary = db_y_summary0_df[
                min_seqs_pass & j_max_pass].reset_index(drop=True)
            df_run[0]["snp_min_seqs_j_max_pass"] = len(db_y_summary)

            if len(db_y_summary) < 1:
                msgs = []
                if min_seqs_pass.sum() == 0:
                    msgs.append(
                        "Not enough sequences (maximum total count is "
                        f"{int(db_y_summary0_df['TOTAL_COUNT'].max())}).")
                if j_max_pass.sum() == 0:
                    pct = round(
                        100 * db_y_summary0_df["MAX_FRAC"].max(), 1)
                    msgs.append(
                        f"A J-junction combination is too prevalent "
                        f"({pct}% of sequences).")
                msg = " and ".join(msgs)
                df_run[0]["note"] = (
                    "Position(s) passed y-intercept ("
                    + ",".join(str(p) for p in pass_y)
                    + ") but " + msg + ".")
                df_run[0]["perfect_match_count"] = int(
                    db_y_summary0_df["TOTAL_COUNT"].max())
                df_run[0]["perfect_match_freq"] = (
                    df_run[0]["perfect_match_count"]
                    / df_run[0]["germline_call_count"])
                if last_iter:
                    break
                continue

            # Novel allele(s) found.
            germ_nts = list(gl_substring)
            for r in range(len(db_y_summary)):
                if r > 0:
                    df_run = [dict(df_run[0])] + df_run
                snp_str = db_y_summary["SNP_STRING"].iloc[r]
                snp_nts = list(snp_str)
                remain_mut = sorted(set(
                    get_mutated_positions([snp_str], [gl_substring])[0]))
                new_germ = insert_polymorphisms(germ, pass_y, snp_nts)
                # Is this a known allele?
                known = [n for n, gseq in germlines.items()
                         if gseq == new_germ]
                if not known:
                    name_parts = []
                    for rm in remain_mut:
                        i0 = rm - 1
                        name_parts.append(
                            f"{germ_nts[i0]}{pass_y[i0]}{snp_nts[i0]}")
                    germ_name = (allele_name + "_"
                                 + "_".join(name_parts))
                else:
                    known_sorted = sort_alleles(known, method="position")
                    germ_name = known_sorted[0]
                df_run[0]["polymorphism_call"] = germ_name
                df_run[0]["novel_imgt"] = new_germ
                df_run[0]["perfect_match_count"] = int(
                    db_y_summary["TOTAL_COUNT"].iloc[r])
                df_run[0]["perfect_match_freq"] = (
                    df_run[0]["perfect_match_count"]
                    / df_run[0]["germline_call_count"])
                df_run[0]["note"] = "Novel allele found!"

        out_frames.extend(df_run)

    out_df = pd.DataFrame(out_frames, columns=_NOVEL_COLUMNS)
    # Object dtype for columns that may hold strings post-fill.
    for _c in ("nt_substitutions", "polymorphism_call", "novel_imgt"):
        out_df[_c] = out_df[_c].astype(object)

    # Post-processing: counts that need the full dataset.
    data_seq_full = data[seq].astype(str)

    def _strip_gaps(s):
        return re.sub(r"[-\.]", "", s)

    data_seq_sub = [_strip_gaps(s[pos_min_r - 1:pos_max_r])
                    for s in data_seq_full]

    def _db_match(novel_imgt_list):
        res = []
        for n in novel_imgt_list:
            n2 = _strip_gaps(str(n)[pos_min_r - 1:pos_max_r])
            res.append(sum(1 for d in data_seq_sub if n2 in d))
        return res

    def _num_j(novel_imgt_list):
        j_genes = get_gene(list(data[j_call]))
        res = []
        for n in novel_imgt_list:
            n2 = _strip_gaps(str(n)[pos_min_r - 1:pos_max_r])
            jset = set()
            for i, d in enumerate(data_seq_sub):
                if n2 in d:
                    jset.add(j_genes[i])
            res.append(len(jset))
        return res

    def _num_cdr3(novel_imgt_list):
        res = []
        junctions = list(data[junction]) if junction in data.columns else None
        for n in novel_imgt_list:
            n2 = _strip_gaps(str(n)[pos_min_r - 1:pos_max_r])
            cset = set()
            for i, d in enumerate(data_seq_sub):
                if n2 in d:
                    jseq = str(junctions[i])
                    cset.add(jseq[3:len(jseq) - 3])
            res.append(len(cset))
        return res

    idx = out_df.index[out_df["novel_imgt"].notna()].tolist()
    if idx:
        out_df.loc[idx, "nt_substitutions"] = [
            _mu_spec(out_df.loc[i, "polymorphism_call"],
                     out_df.loc[i, "germline_call"]) for i in idx]
        out_df.loc[idx, "novel_imgt_count"] = _db_match(
            out_df.loc[idx, "novel_imgt"].tolist())
        out_df.loc[idx, "novel_imgt_unique_j"] = _num_j(
            out_df.loc[idx, "novel_imgt"].tolist())
        if junction in data.columns:
            out_df.loc[idx, "novel_imgt_unique_cdr3"] = _num_cdr3(
                out_df.loc[idx, "novel_imgt"].tolist())

    out_df["germline_imgt_count"] = _db_match(
        out_df["germline_imgt"].tolist())
    out_df["unmutated_freq"] = (out_df["unmutated_count"]
                                / out_df["germline_call_count"])

    # Duplicated novel_imgt handling.
    novel_imgt = out_df["novel_imgt"]
    dup_mask = novel_imgt.duplicated() & out_df["polymorphism_call"].notna()
    dup_values = set(novel_imgt[dup_mask].dropna())
    dup_idx = out_df.index[novel_imgt.isin(dup_values)
                           & novel_imgt.notna()].tolist()
    for i in dup_idx:
        this_poly = out_df.loc[i, "polymorphism_call"]
        this_imgt = out_df.loc[i, "novel_imgt"]
        others = out_df.index[out_df["novel_imgt"] == this_imgt].tolist()
        other_polys = [out_df.loc[o, "polymorphism_call"] for o in others
                       if out_df.loc[o, "polymorphism_call"] != this_poly]
        new_note = ", ".join(str(p) for p in other_polys)
        out_df.loc[i, "note"] = (
            str(out_df.loc[i, "note"]) + ". Same as: " + new_note)

    return out_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
def select_novel(novel: pd.DataFrame, keep_alleles: bool = False) -> pd.DataFrame:
    """Select only the rows of a findNovelAlleles result with novel alleles.

    Port of ``tigger::selectNovel``.
    """
    df = novel[novel["novel_imgt"].notna()].copy()
    if keep_alleles:
        df = df.groupby("germline_call", group_keys=False, sort=False).apply(
            lambda g: g.drop_duplicates(subset="novel_imgt", keep="first"))
    else:
        df = df.drop_duplicates(subset="novel_imgt", keep="first")
    return df.reset_index(drop=True)
