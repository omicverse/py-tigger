"""Sequence and IO utilities — faithful port of tigger's sequence helpers.

Covers ``cleanSeqs``, ``readIgFasta``, ``writeFasta``, ``updateAlleleNames``,
``sortAlleles``, ``getMutatedPositions``, ``getMutCount``,
``findUnmutatedCalls``, ``insertPolymorphisms`` and the private
``superSubstring``.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Union

from .segments import get_allele, get_family, get_gene

__all__ = [
    "clean_seqs",
    "read_ig_fasta",
    "write_fasta",
    "update_allele_names",
    "sort_alleles",
    "get_mutated_positions",
    "get_mut_count",
    "find_unmutated_calls",
    "insert_polymorphisms",
    "super_substring",
]

_NOT_NT = re.compile(r"[^ACGT\.\-]")


# ---------------------------------------------------------------------------
def clean_seqs(seqs):
    """Capitalize nucleotides and replace non-``ACGT.-`` characters with ``N``.

    Port of ``tigger::cleanSeqs``.  Accepts a string, a list of strings, or
    a ``dict`` mapping name -> sequence (named vector).
    """
    if isinstance(seqs, str):
        return _NOT_NT.sub("N", seqs.upper())
    if isinstance(seqs, dict):
        return {k: _NOT_NT.sub("N", str(v).upper()) for k, v in seqs.items()}
    return [_NOT_NT.sub("N", str(s).upper()) for s in seqs]


# ---------------------------------------------------------------------------
def read_ig_fasta(fasta_file, strip_down_name: bool = True,
                  force_caps: bool = True) -> Dict[str, str]:
    """Read a FASTA file of immunoglobulin sequences.

    Port of ``tigger::readIgFasta``.  Returns an ordered ``dict`` mapping
    sequence name -> sequence.
    """
    with open(fasta_file, "r") as fh:
        text = fh.read()
    out: Dict[str, str] = {}
    # Split on '>' that may be preceded by whitespace.
    records = re.split(r"[ \t\r\n\v\f]?>", text)
    for rec in records:
        rec = rec.strip()
        if not rec:
            continue
        lines = rec.splitlines()
        name = lines[0].strip()
        seq = "".join(l.strip() for l in lines[1:])
        seq = re.sub(r"[ \t\r\n\v\f]", "", seq)
        if not seq:
            continue
        if force_caps:
            seq = seq.upper()
        if strip_down_name:
            name = get_allele(name, strip_d=False)
        out[name] = seq
    return out


def write_fasta(named_sequences: Dict[str, str], file, width: int = 60,
                append: bool = False) -> None:
    """Write a named mapping of sequences to a FASTA file.

    Port of ``tigger::writeFasta``.
    """
    mode = "a" if append else "w"
    chunks: List[str] = []
    for name, seq in named_sequences.items():
        seq = str(seq)
        if isinstance(width, int) and 0 < width < 256:
            seq = "\n".join(
                seq[i:i + width] for i in range(0, len(seq), width)
            )
        chunks.append(f">{name}\n{seq}\n")
    with open(file, mode) as fh:
        fh.write("".join(chunks))


# ---------------------------------------------------------------------------
_TEMP_NAMES = [
    "IGHV1-c*", "IGHV1-f*", "IGHV3-d*", "IGHV3-h*",
    "IGHV4-b*", "IGHV5-a*", "IGHV2-5*10", "IGHV2-5*07",
]
_DEF_NAMES = [
    "IGHV1-38-4*", "IGHV1-69-2*", "IGHV3-38-3*", "IGHV3-69-1*",
    "IGHV4-38-2*", "IGHV5-10-1*", "IGHV2-5*02", "IGHV2-5*04",
]


def update_allele_names(allele_calls):
    """Replace outdated IGHV allele names with current IMGT names.

    Port of ``tigger::updateAlleleNames``.
    """
    scalar = isinstance(allele_calls, str)
    calls = [allele_calls] if scalar else list(allele_calls)
    out = []
    for c in calls:
        s = c
        for old, new in zip(_TEMP_NAMES, _DEF_NAMES):
            s = s.replace(old, new)
        out.append(s)
    return out[0] if scalar else out


def sort_alleles(allele_calls, method: str = "name"):
    """Sort Ig allele names by family, gene, then allele.

    Port of ``tigger::sortAlleles``.  ``method`` is ``"name"`` (lexicographic
    locus order) or ``"position"`` (descending locus position).
    """
    if method.startswith("pos"):
        method = "position"
    elif method.startswith("nam"):
        method = "name"
    if method not in ("name", "position"):
        raise ValueError("method must be 'name' or 'position'")

    calls = list(allele_calls)
    submitted = get_allele(calls, first=False, strip_d=False)

    rows = []
    for orig, sub in zip(calls, submitted):
        family = get_family(sub)
        gene = get_gene(sub)
        # GENE1: gsub("[^-]+([-S][^-\\*D]+).*","\\1", sub); sub("^-","")
        m = re.match(r"[^-]+([-S][^-\*D]+).*", sub)
        gene1 = m.group(1) if m else sub
        gene1 = re.sub(r"^-", "", gene1)
        gene1 = re.sub(r"[^0-9]+", "99", gene1)
        try:
            gene1_n = float(gene1) if gene1 != "" else 0.0
        except ValueError:
            gene1_n = 0.0
        # GENE2: gsub("[^-]+[-S][^-]+-?","", gene)
        gene2 = re.sub(r"[^-]+[-S][^-]+-?", "", gene)
        gene2 = re.sub(r"[^0-9]+", "99", gene2)
        try:
            gene2_n = float(gene2) if gene2 != "" else 0.0
        except ValueError:
            gene2_n = 0.0
        # ALLELE — R's sub() replaces only the first match.
        allele = get_allele(sub)
        allele = re.sub(r"[^\*]+\*|[^\*]+$", "", allele, count=1)
        allele = re.sub(r"_.+$", "", allele, count=1)
        try:
            allele_n = float(allele) if allele != "" else 0.0
        except ValueError:
            allele_n = 0.0
        rows.append({
            "SUBMITTED_CALLS": sub, "SUBMITTED_NAMES": orig,
            "FAMILY": family, "GENE1": gene1_n, "GENE2": gene2_n,
            "ALLELE": allele_n,
        })

    # R: arrange(allele_df, SUBMITTED_CALLS) first.
    rows.sort(key=lambda r: r["SUBMITTED_CALLS"])
    if method == "name":
        rows.sort(key=lambda r: (r["FAMILY"], r["GENE1"],
                                 r["GENE2"], r["ALLELE"]))
    else:  # position: descending GENE1, GENE2, FAMILY, ALLELE
        rows.sort(key=lambda r: (r["GENE1"], r["GENE2"],
                                 r["FAMILY"], r["ALLELE"]), reverse=True)
    return [r["SUBMITTED_NAMES"] for r in rows]


# ---------------------------------------------------------------------------
_DEFAULT_IGNORE = re.compile(r"[\.N-]")


def get_mutated_positions(samples, germlines, ignored_regex: str = r"[\.N-]",
                          match_instead: bool = False) -> List[List[int]]:
    """Find 1-based positions where aligned sequences differ.

    Port of ``tigger::getMutatedPositions``.  Returns a list (one element per
    sample) of position lists.  Positions where either sequence has an
    ignored character (gap / ``N`` by default) are excluded.
    """
    if isinstance(samples, str):
        samples = [samples]
    if isinstance(germlines, str):
        germlines = [germlines] * len(samples)
    else:
        germlines = list(germlines)
        if len(germlines) == 1:
            germlines = germlines * len(samples)
    if len(samples) != len(germlines):
        raise ValueError(
            "Number of input sequences does not match number of germlines.")

    ig_pat = (re.compile(ignored_regex)
              if ignored_regex != r"[\.N-]" else _DEFAULT_IGNORE)
    out: List[List[int]] = []
    for samp, germ in zip(samples, germlines):
        samp = str(samp)
        germ = str(germ)
        n = min(len(samp), len(germ))
        s = samp[:n].upper()
        g = germ[:n].upper()
        ignore = set()
        for m in ig_pat.finditer(g):
            ignore.update(range(m.start(), m.end()))
        for m in ig_pat.finditer(s):
            ignore.update(range(m.start(), m.end()))
        positions = []
        for i in range(n):
            if i in ignore:
                continue
            differ = s[i] != g[i]
            if (differ and not match_instead) or (not differ and match_instead):
                positions.append(i + 1)
        out.append(positions)
    return out


def get_mut_count(samples, allele_calls, germline_db: Dict[str, str]):
    """Hamming distance of sequences to each germline allele in their call.

    Port of ``tigger::getMutCount``.  For single-allele calls the element is
    an ``int``; for multi-allele calls it is a ``dict`` allele -> int.
    """
    if isinstance(samples, str):
        samples = [samples]
    if isinstance(allele_calls, str):
        allele_calls = [allele_calls]
    out: List = [None] * len(samples)
    for i, (samp, call) in enumerate(zip(samples, allele_calls)):
        alleles = str(call).split(",")
        germs = [germline_db.get(a) for a in alleles]
        if len(germs) == 1:
            g = germs[0]
            if g is None:
                out[i] = float("inf")
            else:
                out[i] = len(get_mutated_positions([samp], [g])[0])
        else:
            d = {}
            for a, g in zip(alleles, germs):
                if g is None:
                    d[a] = float("inf")
                else:
                    d[a] = len(get_mutated_positions([g], [samp])[0])
            out[i] = d
    return out


def _flatten_counts(x):
    """Yield all int distances from a get_mut_count element."""
    if isinstance(x, dict):
        for v in x.values():
            yield v
    else:
        yield x


def find_unmutated_calls(allele_calls, sample_seqs,
                         germline_db: Dict[str, str]) -> List[str]:
    """Return the subset of allele calls representing unmutated sequences.

    Port of ``tigger::findUnmutatedCalls``.
    """
    allele_calls = get_allele(list(allele_calls), first=False)
    sample_seqs = [str(s) for s in sample_seqs]

    germ_names = set(germline_db.keys())
    # Identify calls (rows) referencing alleles not in germline_db.
    keep_idx = []
    for i, call in enumerate(allele_calls):
        if call == "":
            continue
        parts = call.split(",")
        if all(p in germ_names for p in parts):
            keep_idx.append(i)
    calls = [allele_calls[i] for i in keep_idx]
    seqs = [sample_seqs[i] for i in keep_idx]

    mut_counts = get_mut_count(seqs, calls, germline_db)

    out: List[str] = []
    for call, mc in zip(calls, mut_counts):
        dists = list(_flatten_counts(mc))
        if min(dists) != 0:
            continue  # sequence is mutated against all alleles
        split = call.split(",")
        if isinstance(mc, dict):
            # which alleles are perfect matches, in the order of the call
            unmut = [a for a in split if mc.get(a) == 0]
        else:
            unmut = [split[j] for j, d in enumerate([mc]) if d == 0]
        out.append(",".join(unmut))
    return out


# ---------------------------------------------------------------------------
def insert_polymorphisms(sequence: str, positions, nucleotides) -> str:
    """Replace nucleotides at the given 1-based positions.

    Port of ``tigger::insertPolymorphisms``.
    """
    positions = list(positions)
    nucleotides = list(nucleotides)
    if len(positions) != len(nucleotides):
        raise ValueError(
            "Number of nucleotides and number of positions do not match.")
    chars = list(sequence)
    for pos, nt in zip(positions, nucleotides):
        idx = pos - 1
        if 0 <= idx < len(chars):
            chars[idx] = nt
    return "".join(chars)


def super_substring(string: str, positions) -> str:
    """Extract characters at the given 1-based positions (possibly disjoint).

    Port of tigger's private ``superSubstring``.
    """
    chars = []
    for p in positions:
        idx = p - 1
        chars.append(string[idx] if 0 <= idx < len(string) else "")
    return "".join(chars)
