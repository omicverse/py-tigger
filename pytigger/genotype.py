"""Genotype inference and allele reassignment — faithful port of tigger.

Covers ``inferGenotype`` (frequency method), ``inferGenotypeBayesian``
(Dirichlet-multinomial Bayesian model), ``genotypeFasta`` and
``reassignAlleles``.
"""
from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from typing import Dict, List

import numpy as np
import pandas as pd

from .segments import get_allele, get_family, get_gene
from .sequences import (
    find_unmutated_calls,
    get_mut_count,
    get_mutated_positions,
    sort_alleles,
)

__all__ = [
    "infer_genotype",
    "infer_genotype_bayesian",
    "genotype_fasta",
    "reassign_alleles",
]


def _strip_to_allele(name: str) -> str:
    """R: gsub('[^d\\*]*[d\\*]','',name) — strip everything up to and
    including the gene/duplication marker, leaving the allele part."""
    return re.sub(r"[^d\*]*[d\*]", "", name)


# ---------------------------------------------------------------------------
def _prepare_allele_calls(data, v_call, seq, germline_db, novel,
                          find_unmutated):
    """Shared allele-call preparation for both genotype inference methods."""
    allele_calls = get_allele(list(data[v_call]), first=False, strip_d=False)
    germline_db = dict(germline_db) if germline_db is not None else {}

    if find_unmutated:
        if not germline_db:
            raise ValueError("germline_db needed if find_unmutated is TRUE")
        if novel is not None and isinstance(novel, pd.DataFrame):
            nv = novel[novel["polymorphism_call"].notna()][
                ["germline_call", "polymorphism_call", "novel_imgt"]]
            if len(nv) > 0:
                for _, row in nv.iterrows():
                    germline_db[row["polymorphism_call"]] = row["novel_imgt"]
                for _, row in nv.iterrows():
                    gc = row["germline_call"]
                    pc = row["polymorphism_call"]
                    for i, ac in enumerate(allele_calls):
                        if gc in ac:
                            allele_calls[i] = ac + "," + pc
        allele_calls = find_unmutated_calls(
            allele_calls, [str(s) for s in data[seq]], germline_db)
        if len(allele_calls) == 0:
            raise ValueError(
                "No unmutated sequences found! Set 'find_unmutated' to "
                "'FALSE'.")
    return allele_calls, germline_db


def _build_gene_groups(allele_calls, cutoff=None):
    """Group allele-call row indices by gene (R: gene_groups)."""
    all_genes = []
    for ac in allele_calls:
        for piece in ac.split(","):
            all_genes.append(get_gene(piece, strip_d=False))
    seen = []
    for g in all_genes:
        if g not in seen:
            seen.append(g)
    gene_regex = [g + "*" for g in seen]
    gene_groups = OrderedDict()
    for gr in gene_regex:
        gname = gr[:-1]  # remove the literal '*'
        idx = [i for i, ac in enumerate(allele_calls) if gr in ac]
        gene_groups[gname] = idx
    if cutoff is not None:
        gene_groups = OrderedDict(
            (k, v) for k, v in gene_groups.items() if len(v) >= cutoff)
    ordered = sort_alleles(list(gene_groups.keys()))
    return OrderedDict((k, gene_groups[k]) for k in ordered)


# ---------------------------------------------------------------------------
def infer_genotype(data: pd.DataFrame, germline_db: Dict[str, str] = None,
                   novel: pd.DataFrame = None, v_call: str = "v_call",
                   seq: str = "sequence_alignment",
                   fraction_to_explain: float = 0.875,
                   gene_cutoff: float = 1e-4,
                   find_unmutated: bool = True) -> pd.DataFrame:
    """Infer a subject's V genotype using the frequency method.

    Faithful port of ``tigger::inferGenotype``.
    """
    for c in (v_call, seq):
        if c not in data.columns:
            raise ValueError(f"Column not found: {c}")

    allele_calls, germline_db = _prepare_allele_calls(
        data, v_call, seq, germline_db, novel, find_unmutated)

    cutoff = (len(allele_calls) * gene_cutoff
              if gene_cutoff < 1 else gene_cutoff)
    gene_groups = _build_gene_groups(allele_calls, cutoff=cutoff)

    genes = list(gene_groups.keys())
    rows = []
    for g in genes:
        total = len(gene_groups[g])
        # Allele calls restricted to this gene.
        ac = []
        for i in gene_groups[g]:
            pieces = allele_calls[i].split(",")
            kept = [p for p in pieces if re.search(re.escape(g) + r"\*", p)]
            ac.append(",".join(kept))
        target = math.ceil(fraction_to_explain * len(ac))
        t_ac = Counter(ac)
        potentials = []
        for name in t_ac:
            for p in name.split(","):
                if p not in potentials:
                    potentials.append(p)

        if len(potentials) == 1 or len(t_ac) == 1:
            allele = _strip_to_allele(potentials[0])
            count = str(list(t_ac.values())[0])
            rows.append({"gene": g, "alleles": allele, "counts": count,
                         "total": total, "note": ""})
        else:
            # Table of which alleles explain which calls.
            call_names = list(t_ac.keys())
            call_counts = np.array([t_ac[c] for c in call_names], dtype=float)
            # regex per potential: "pot$" or "pot," (matches in the call)
            seqs_expl = np.zeros((len(call_names), len(potentials)))
            for j, pot in enumerate(potentials):
                esc = re.escape(pot)
                pat = re.compile(esc + r"$|" + esc + r",")
                for i, cn in enumerate(call_names):
                    if pat.search(cn):
                        seqs_expl[i, j] = call_counts[i]
            included = []
            counts_out = []
            tot_expl = 0.0
            active = np.ones(len(call_names), dtype=bool)
            while tot_expl < target:
                allele_tot = (seqs_expl * active[:, None]).sum(axis=0)
                best_j = int(np.argmax(allele_tot))
                best_val = allele_tot[best_j]
                included.append(potentials[best_j])
                counts_out.append(int(best_val))
                tot_expl += best_val
                # Drop rows already explained by best allele.
                explained = seqs_expl[:, best_j] != 0
                active = active & (~explained)
            rows.append({
                "gene": g,
                "alleles": ",".join(_strip_to_allele(a) for a in included),
                "counts": ",".join(str(c) for c in counts_out),
                "total": total, "note": ""})

    geno = pd.DataFrame(rows, columns=["gene", "alleles", "counts",
                                       "total", "note"])

    # Check for indistinguishable calls.
    if find_unmutated:
        geno = _annotate_indistinguishable(geno, germline_db)
    return geno.reset_index(drop=True)


def _annotate_indistinguishable(geno, germline_db):
    """Add 'Cannot distinguish' notes for genotype alleles with 0 distance."""
    seqs = genotype_fasta(geno, germline_db)
    names = list(seqs.keys())
    n = len(names)
    if n == 0:
        return geno
    dist = np.full((n, n), np.nan)
    seq_list = [seqs[nm] for nm in names]
    for j, nm in enumerate(names):
        col = get_mutated_positions(seq_list, seqs[nm])
        for i in range(n):
            dist[i, j] = len(col[i])
    for i in range(n):
        dist[i, i] = np.nan
    # R: which(dist_mat == 0, arr.ind=TRUE) iterates column-major.
    same = [(i, j) for j in range(n) for i in range(n) if dist[i, j] == 0]
    for (i, j) in same:
        gene_i = get_gene(names[i])
        msg = "Cannot distinguish " + names[i] + " and " + names[j]
        mask = geno["gene"] == gene_i
        geno.loc[mask, "note"] = msg
    return geno


# ---------------------------------------------------------------------------
def _ddirichlet(x, alpha):
    """Dirichlet density — port of gtools::ddirichlet (single observation)."""
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    # logD = sum(lgamma(alpha)) - lgamma(sum(alpha))
    log_d = np.sum([math.lgamma(a) for a in alpha]) - math.lgamma(alpha.sum())
    # s = sum((alpha - 1) * log(x)); R's ddirichlet sets log(0)*?=... :
    # gtools uses: logterm = sum((alpha-1)*log(x)); if any x==0 & alpha>1 -> 0
    with np.errstate(divide="ignore"):
        logx = np.log(x)
    s = 0.0
    valid = True
    for xi, ai, lxi in zip(x, alpha, logx):
        if xi == 0:
            if ai > 1:
                s += (ai - 1) * lxi  # -> -inf
            elif ai < 1:
                s += (ai - 1) * lxi  # -> +inf
            # ai == 1: term is 0
        else:
            s += (ai - 1) * lxi
    if not np.isfinite(s):
        # gtools ddirichlet: returns 0 when x has a zero where it shouldn't
        if s == float("-inf"):
            return 0.0
        return math.exp(s - log_d) if np.isfinite(s) else float("inf")
    return math.exp(s - log_d)


def _get_probabilities_with_priors(X, alpha_dirichlet=None, epsilon=0.01,
                                   priors=None):
    """Port of tigger's private ``get_probabilities_with_priors``."""
    if alpha_dirichlet is None:
        alpha_dirichlet = np.array([0.5, 0.5, 0.5, 0.5]) * 2
    if priors is None:
        priors = [0.5, 0.5, 0.33, 0.33, 0.33, 0.25, 0.25, 0.25, 0.25]
    X = np.array(sorted(X, reverse=True), dtype=float)
    alpha_dirichlet = np.asarray(alpha_dirichlet, dtype=float)

    H1 = np.array([1.0, 0, 0, 0])
    H2 = np.array([priors[0], priors[1], 0, 0])
    H3 = np.array([priors[2], priors[3], priors[4], 0])
    H4 = np.array([priors[5], priors[6], priors[7], priors[8]])

    def _e(H, x):
        p = (H + epsilon) / np.sum(H + epsilon)
        return _ddirichlet(p, alpha_dirichlet + x)

    E1 = _e(H1, X)
    E2 = _e(H2, X)
    E3 = _e(H3, X)
    E4 = _e(H4, X)

    # While the 2nd-largest is exactly 0, scale X down by 10.
    guard = 0
    while sorted([E1, E2, E3, E4], reverse=True)[1] == 0 and guard < 1000:
        X = X / 10.0
        E1 = _e(H1, X)
        E2 = _e(H2, X)
        E3 = _e(H3, X)
        E4 = _e(H4, X)
        guard += 1

    with np.errstate(divide="ignore"):
        return np.log10(np.array([E1, E2, E3, E4]))


def infer_genotype_bayesian(data: pd.DataFrame,
                            germline_db: Dict[str, str] = None,
                            novel: pd.DataFrame = None, v_call: str = "v_call",
                            seq: str = "sequence_alignment",
                            find_unmutated: bool = True,
                            priors=None) -> pd.DataFrame:
    """Infer a subject's V genotype using a Bayesian Dirichlet-multinomial.

    Faithful port of ``tigger::inferGenotypeBayesian``.
    """
    if priors is None:
        priors = [0.6, 0.4, 0.4, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25]

    allele_calls, germline_db = _prepare_allele_calls(
        data, v_call, seq, germline_db, novel, find_unmutated)

    gene_groups = _build_gene_groups(allele_calls, cutoff=None)
    genes = list(gene_groups.keys())

    rows = []
    for g in genes:
        total = len(gene_groups[g])
        ac = []
        for i in gene_groups[g]:
            pieces = allele_calls[i].split(",")
            kept = [p for p in pieces if re.search(re.escape(g) + r"\*", p)]
            ac.append(",".join(kept))
        t_ac = Counter(ac)
        call_names = list(t_ac.keys())
        potentials = []
        for name in call_names:
            for p in name.split(","):
                if p not in potentials:
                    potentials.append(p)

        # seqs_expl matrix: rows = call_names, cols = potentials.
        expl = {}  # call_name -> {potential: count}
        for cn in call_names:
            esc_map = {}
            for pot in potentials:
                esc = re.escape(pot)
                pat = re.compile(esc + r"$|" + esc + r",")
                esc_map[pot] = (t_ac[cn] if pat.search(cn) else 0.0)
            expl[cn] = esc_map

        # Add fake low counts for potentials with no own single-allele row.
        for pot in potentials:
            if pot not in expl:
                expl[pot] = {p: 0.0 for p in potentials}
                expl[pot][pot] = 0.01

        # Split single vs multi (comma) rows.
        single_rows = OrderedDict(
            (k, v) for k, v in expl.items() if "," not in k)
        multi_rows = OrderedDict(
            (k, v) for k, v in expl.items() if "," in k)

        # Distribute multi-assigned reads ratio-dependently.
        if single_rows and 0 < len(single_rows) < len(expl) and multi_rows:
            multi_keys = sorted(multi_rows.keys(), key=len)
            for mk in multi_keys:
                m_genes = mk.split(",")
                # counts = colSums of single rows that are among m_genes,
                # restricted to columns m_genes.
                counts = {gg: 0.0 for gg in m_genes}
                for srk, srv in single_rows.items():
                    if srk in m_genes:
                        for gg in m_genes:
                            counts[gg] += srv.get(gg, 0.0)
                cdist = {gg: multi_rows[mk].get(gg, 0.0) for gg in m_genes}
                tot = sum(counts.values())
                if tot == 0:
                    continue
                for gg in m_genes:
                    new_c = (counts[gg]
                             + (cdist[gg] * counts[gg]) / tot)
                    if gg in single_rows:
                        single_rows[gg][gg] = new_c

        rows_used = single_rows if single_rows else expl
        # Round and drop all-zero rows.
        mat = {}
        for rk, rv in rows_used.items():
            mat[rk] = {p: round(rv.get(p, 0.0)) for p in potentials}
        mat = {rk: rv for rk, rv in mat.items() if sum(rv.values()) != 0}

        # allele_tot: column sums, sorted descending.
        col_tot = {p: 0.0 for p in potentials}
        for rv in mat.values():
            for p in potentials:
                col_tot[p] += rv[p]
        allele_tot = sorted(col_tot.items(), key=lambda kv: kv[1],
                            reverse=True)
        allele_names = [a for a, _ in allele_tot]
        allele_vals = [v for _, v in allele_tot]

        length = min(len(allele_vals), 4)
        x_in = sorted(allele_vals + [0] * (4 - length), reverse=True)[:4]
        probs = _get_probabilities_with_priors(x_in, priors=priors)
        probs = np.where(probs == float("-inf"), -1000.0, probs)
        k_sorted = sorted(probs.tolist(), reverse=True)
        k_diff = k_sorted[0] - k_sorted[1]

        rows.append({
            "gene": g,
            "alleles": ",".join(
                _strip_to_allele(a) for a in allele_names[:length]),
            "counts": ",".join(
                str(int(v)) for v in allele_vals[:length]),
            "total": total, "note": "",
            "kh": probs[0], "kd": probs[1], "kt": probs[2],
            "kq": probs[3], "k_diff": k_diff,
        })

    geno = pd.DataFrame(rows, columns=["gene", "alleles", "counts", "total",
                                       "note", "kh", "kd", "kt", "kq",
                                       "k_diff"])
    if find_unmutated:
        geno = _annotate_indistinguishable(geno, germline_db)
    return geno.reset_index(drop=True)


# ---------------------------------------------------------------------------
def genotype_fasta(genotype: pd.DataFrame, germline_db: Dict[str, str],
                   novel: pd.DataFrame = None) -> "OrderedDict[str, str]":
    """Return the germline nucleotide sequences for a genotype.

    Faithful port of ``tigger::genotypeFasta``.
    """
    germline_db = dict(germline_db)
    if novel is not None and isinstance(novel, pd.DataFrame):
        nv = novel[novel["polymorphism_call"].notna()][
            ["germline_call", "polymorphism_call", "novel_imgt"]]
        for _, row in nv.iterrows():
            germline_db[row["polymorphism_call"]] = row["novel_imgt"]

    genes = [get_gene(g, first=True, strip_d=True) for g in genotype["gene"]]

    # g_names: list of (key, real_name) where key = getAllele(strip_d=TRUE).
    # R: names(g_names) <- getAllele(...); duplicates are *kept*.
    g_names = [(get_allele(name, first=True, strip_d=True), name)
               for name in germline_db]
    g_keys = {k for k, _ in g_names}

    table_calls = []
    for gene, alleles in zip(genes, genotype["alleles"]):
        for a in str(alleles).split(","):
            table_calls.append(gene + "*" + a)

    not_found = [tc for tc in table_calls if tc not in g_keys]
    if not_found:
        raise ValueError(
            "The following genotype alleles were not found in germline_db: "
            + ", ".join(not_found))

    # R: g_names[names(g_names) %in% table_calls_names] — returns *all*
    # germline entries whose stripped key is in table_calls (D-duplicates
    # included), in germline_db order.
    table_set = set(table_calls)
    out = OrderedDict()
    for key, real_name in g_names:
        if key in table_set:
            out[real_name] = germline_db[real_name]
    return out


# ---------------------------------------------------------------------------
def reassign_alleles(data: pd.DataFrame, genotype_db: Dict[str, str],
                     v_call: str = "v_call",
                     seq: str = "sequence_alignment",
                     method: str = "hamming", path=None,
                     keep_gene: str = "gene") -> pd.DataFrame:
    """Correct V-call assignments using a personalized genotype.

    Faithful port of ``tigger::reassignAlleles``.  Adds a
    ``v_call_genotyped`` column to a copy of ``data``.
    """
    if keep_gene not in ("gene", "family", "repertoire"):
        raise ValueError(f"Unknown keep_gene value: {keep_gene}")
    if method != "hamming":
        raise ValueError(
            "Only Hamming distance is currently supported as a method.")

    genotype_db = OrderedDict(genotype_db)
    v_sequences = [str(s) for s in data[seq]]
    v_calls = get_allele(list(data[v_call]), first=False, strip_d=False)
    n = len(v_calls)
    v_call_genotyped = [""] * n

    geno_names = list(genotype_db.keys())
    if keep_gene == "gene":
        v = get_gene(v_calls, first=True, strip_d=False)
        geno = {nm: get_gene(nm, strip_d=True) for nm in geno_names}
    elif keep_gene == "family":
        v = get_family(v_calls, first=True, strip_d=False)
        geno = {nm: get_family(nm, strip_d=True) for nm in geno_names}
    else:  # repertoire
        v = [v_call] * n
        geno = {nm: v_call for nm in geno_names}

    # geno mapped value list, in genotype_db order.
    geno_vals = [geno[nm] for nm in geno_names]

    # Homozygous vs heterozygous genes/families.
    seen_dup = []
    counts_seen = Counter(geno_vals)
    hetero = [g for g in dict.fromkeys(geno_vals) if counts_seen[g] > 1]
    homo = {nm: geno[nm] for nm in geno_names if geno[nm] not in hetero}
    # homo_alleles: gene/family -> allele name
    homo_alleles = {}
    for nm, gval in homo.items():
        homo_alleles[gval] = nm
    homo_genes = set(homo.values())
    for i in range(n):
        if v[i] in homo_genes:
            v_call_genotyped[i] = homo_alleles[v[i]]

    homo_calls_i = [i for i in range(n) if v[i] in homo_genes]

    # Heterozygous: realign each sequence to each allele.
    for het in hetero:
        ind = [i for i in range(n) if v[i] == het]
        if not ind:
            continue
        het_alleles = [nm for nm in geno_names if geno[nm] == het]
        het_seqs = [genotype_db[a] for a in het_alleles]
        # dist_mat: rows = sequences, cols = het alleles.
        dist_cols = []
        for hs in het_seqs:
            d = [len(p) for p in get_mutated_positions(
                [v_sequences[i] for i in ind], hs, match_instead=False)]
            dist_cols.append(d)
        dist_mat = np.array(dist_cols).T  # shape (len(ind), len(het_alleles))
        for r, i in enumerate(ind):
            row = dist_mat[r]
            mn = row.min()
            best = [het_alleles[j] for j in range(len(row)) if row[j] == mn]
            v_call_genotyped[i] = ",".join(best)

    # Genes not in genotype: realign to every genotype allele.
    hetero_calls_i = [i for i in range(n) if v[i] in hetero]
    assigned = set(homo_calls_i) | set(hetero_calls_i)
    not_called = [i for i in range(n) if i not in assigned]
    if len(not_called) > 1:
        all_seqs = list(genotype_db.values())
        all_names = list(genotype_db.keys())
        dist_cols = []
        for gs in all_seqs:
            d = [len(p) for p in get_mutated_positions(
                [v_sequences[i] for i in not_called], gs,
                match_instead=False)]
            dist_cols.append(d)
        dist_mat = np.array(dist_cols).T
        for r, i in enumerate(not_called):
            row = dist_mat[r]
            mn = row.min()
            best = [all_names[j] for j in range(len(row)) if row[j] == mn]
            v_call_genotyped[i] = ",".join(best)

    out = data.copy()
    out["v_call_genotyped"] = v_call_genotyped

    if all(v_call_genotyped[i] == str(data[v_call].iloc[i])
           for i in range(n)):
        import warnings
        msg = "No allele assignment corrections made."
        warnings.warn(msg)

    return out
