"""Evidence-table generation — faithful port of tigger's ``generateEvidence``.

Builds a table of evidence metrics for final novel-allele detection and
genotyping inferences, combining a reassigned data frame, the novel-allele
table, the genotype, and the germline databases.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import pandas as pd

from .segments import get_allele, get_family, get_gene, translate_dna
from .sequences import clean_seqs, get_mut_count, get_mutated_positions

__all__ = ["generate_evidence", "has_non_imgt_gaps", "get_mutated_aa"]


def has_non_imgt_gaps(seq: str) -> bool:
    """Return ``True`` if a sequence has non-triplet gaps.

    Port of tigger's private ``hasNonImgtGaps``.
    """
    length = math.ceil(len(seq) / 3) * 3
    codons = [seq[i:i + 3] for i in range(0, length, 3)]
    for codon in codons:
        gap_len = len(re.sub(r"[^\.\-]", "", codon))
        if gap_len % 3 != 0:
            return True
    return False


def get_mutated_aa(ref_imgt: str, novel_imgt: str) -> List[str]:
    """Compare two IMGT-gapped sequences and list amino-acid mutations.

    Port of tigger's private ``getMutatedAA``.
    """
    ref = list(translate_dna(ref_imgt))
    novel = list(translate_dna(novel_imgt))
    mutations = []
    n = min(len(ref), len(novel))
    for i in range(n):
        if ref[i] != novel[i]:
            nt = novel[i] if novel[i] is not None else "-"
            mutations.append(f"{i + 1}{ref[i]}>{nt}")
    return mutations


def _find_closest_reference(seq_name, seq_val, allele_calls, ref_germ,
                            exclude_self=False, multiple=False):
    """Port of generateEvidence's private ``.findClosestReference``."""
    ref_germ = dict(ref_germ)
    allele_calls = list(allele_calls)

    seq_fam = re.sub(r"[0-9]+$", "", get_family(seq_name))
    # Filter ref_germ: keep only same gene segment as seq.
    diff_calls = {nm for nm in ref_germ
                  if re.sub(r"[0-9]+$", "", get_family(nm)) != seq_fam}
    if diff_calls:
        allele_calls = [a for a in allele_calls if a not in diff_calls]
        ref_germ = {k: v for k, v in ref_germ.items() if k not in diff_calls}

    closest = get_mut_count([seq_val], ",".join(allele_calls), ref_germ)[0]
    if isinstance(closest, dict):
        dist_items = list(closest.items())
    else:
        dist_items = [(allele_calls[0], closest)]
    dist_vals = [d for _, d in dist_items]
    min_dist = min(dist_vals)
    closest_names = []
    for a, d in zip([allele_calls[i] for i in range(len(dist_items))],
                    dist_vals):
        if d == min_dist and a not in closest_names:
            closest_names.append(a)

    if exclude_self and seq_name in closest_names:
        closest_names = [c for c in closest_names if c != seq_name]

    if len(closest_names) > 1:
        # Keep ones with fewest mutated positions (count of '_').
        mut_pos_count = {c: len(re.sub(r"[^_]", "", c))
                         for c in closest_names}
        mn = min(mut_pos_count.values())
        closest_names = [c for c in closest_names
                         if mut_pos_count[c] == mn]
        # Pick same length.
        if len(closest_names) > 1 and seq_name in ref_germ:
            target_len = len(ref_germ[seq_name])
            same_len = [c for c in closest_names
                        if c in ref_germ and len(ref_germ[c]) == target_len]
            if same_len:
                closest_names = same_len
        # Pick same allele.
        if len(closest_names) > 1:
            target_allele = re.sub(r"_.+", "", get_allele(seq_name))
            same_allele = [c for c in closest_names
                           if get_allele(c) == target_allele]
            if same_allele:
                closest_names = same_allele
        # Pick non-duplicated.
        if len(closest_names) > 1:
            non_dup = [c for c in closest_names
                       if not re.search(r"D\*", c)]
            if non_dup:
                closest_names = non_dup
        if len(closest_names) > 1 and not multiple:
            raise ValueError(
                "Multiple closest reference found for " + str(seq_name)
                + ":\n" + ",".join(closest_names))
    return closest_names


def generate_evidence(data: pd.DataFrame, novel: pd.DataFrame,
                      genotype: pd.DataFrame,
                      genotype_db: Dict[str, str],
                      germline_db: Dict[str, str],
                      j_call: str = "j_call", junction: str = "junction",
                      fields: Optional[List[str]] = None) -> pd.DataFrame:
    """Build a table of evidence metrics for novel alleles.

    Faithful port of ``tigger::generateEvidence``.
    """
    genotype_db = dict(genotype_db)
    germline_db = dict(germline_db)

    # germline_set = germline_db (minus genotype names) + genotype_db.
    germline_set = {k: v for k, v in germline_db.items()
                    if k not in genotype_db}
    germline_set.update(genotype_db)

    # Subset genotype to novel alleles.
    novel_polys = set(novel["polymorphism_call"].dropna())
    rows = []
    for _, grow in genotype.iterrows():
        gene = grow["gene"]
        alleles = str(grow["alleles"]).split(",")
        counts = str(grow["counts"]).split(",")
        seen_alleles = set()
        for a, c in zip(alleles, counts):
            if a in seen_alleles:
                continue
            seen_alleles.add(a)
            poly = f"{gene}*{a}"
            if poly in novel_polys:
                rows.append({
                    "gene": gene, "allele": a, "counts": c,
                    "total": grow["total"], "note_gt": grow["note"],
                    "polymorphism_call": poly,
                })
    final_gt = pd.DataFrame(rows)
    if len(final_gt) == 0:
        return final_gt

    # Join with novel by polymorphism_call.
    final_gt = final_gt.merge(novel, on="polymorphism_call", how="inner")

    # Note for novel_imgt with multiple polymorphism calls.
    num_calls = final_gt.groupby("novel_imgt")["polymorphism_call"].transform(
        lambda s: s.nunique())
    idx_mult = final_gt.index[num_calls > 1]
    for i in idx_mult:
        final_gt.loc[i, "note_gt"] = (
            str(final_gt.loc[i, "note_gt"])
            + " Found multiple polymorphism calls for the same novel_imgt.")

    out_rows = []
    v_call_genotyped = list(data["v_call_genotyped"])
    for _, df in final_gt.iterrows():
        rec = dict(df)
        poly = rec["polymorphism_call"]
        novel_imgt = rec["novel_imgt"]

        sequences = sum(1 for x in v_call_genotyped if x == poly)
        rec["sequences"] = sequences

        closest_ref_input = _find_closest_reference(
            poly, novel_imgt, list(germline_db.keys()), germline_db,
            exclude_self=False)[0]
        rec["closest_reference"] = closest_ref_input

        # nt diff vs closest reference.
        ref_seq = germline_set[closest_ref_input]
        nt_diff = list(get_mutated_positions(novel_imgt, ref_seq)[0])
        if len(novel_imgt) < len(ref_seq):
            nt_diff += list(range(len(novel_imgt) + 1, len(ref_seq) + 1))
        nt_diff_string = ""
        if nt_diff:
            ref_chars = list(germline_set[closest_ref_input])
            novel_chars = list(germline_set[poly])
            parts = []
            for p in nt_diff:
                i0 = p - 1
                rc = ref_chars[i0] if i0 < len(ref_chars) else "-"
                nc = novel_chars[i0] if i0 < len(novel_chars) else "-"
                parts.append(f"{p}{rc}>{nc}")
            nt_diff_string = ",".join(parts)
        rec["nt_diff"] = len(nt_diff)
        rec["nt_substitutions"] = nt_diff_string

        diff_aa = get_mutated_aa(germline_set[closest_ref_input],
                                 germline_set[poly])
        rec["aa_diff"] = len(diff_aa)
        rec["aa_substitutions"] = ",".join(diff_aa) if diff_aa else ""

        rec["counts"] = float(rec["counts"])
        rec["total"] = float(rec["total"])
        rec["unmutated_sequences"] = rec["counts"]
        rec["unmutated_frequency"] = (
            rec["counts"] / sequences if sequences else float("nan"))
        rec["allelic_percentage"] = (
            100.0 * rec["unmutated_sequences"] / rec["total"]
            if rec["total"] else float("nan"))

        if sequences > 0:
            sub = data[data["v_call_genotyped"] == poly]
            rec["unique_js"] = sub[j_call].nunique()
            cdr3 = translate_dna(list(sub[junction]), trim=True)
            rec["unique_cdr3s"] = len(set(cdr3))
        else:
            rec["unique_js"] = float("nan")
            rec["unique_cdr3s"] = float("nan")

        rec["closest_reference_imgt"] = clean_seqs(
            germline_set[closest_ref_input])
        out_rows.append(rec)

    result = pd.DataFrame(out_rows)
    result["note"] = (result["note_gt"].fillna("").astype(str) + " "
                      + result["note"].fillna("").astype(str)).str.strip()
    result = result.drop(columns=["note_gt"])
    return result.reset_index(drop=True)
