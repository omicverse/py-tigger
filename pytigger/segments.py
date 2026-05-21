"""Immunoglobulin segment-name parsing — faithful port of alakazam's
``getSegment`` / ``getGene`` / ``getAllele`` / ``getFamily``.

tigger depends on alakazam for these helpers.  The regular expressions and
processing order are reproduced exactly so allele/gene/family extraction
matches the R behaviour bit-for-bit.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Union

__all__ = [
    "get_segment",
    "get_gene",
    "get_allele",
    "get_family",
    "translate_dna",
]

# alakazam regexes (Perl-compatible).  Python's ``re`` understands them
# directly once the doubled backslashes are reduced to single ones.
_GENE_REGEX = r"((IG[HKL]|TR[ABDG])[VDJADEGMC][A-R0-9\(\)]*[-/\w]*)"
_ALLELE_REGEX = (
    r"((IG[HKL]|TR[ABDG])[VDJADEGMC][A-R0-9\(\)]*[-/\w]*[-\*]*[\.\w]+)"
)
_FAMILY_REGEX = r"((IG[HKL]|TR[ABDG])[VDJADEGMC][A-R0-9\(\)]*)"


def _is_str(x) -> bool:
    return isinstance(x, str)


def get_segment(
    segment_call: Union[str, Iterable[str]],
    segment_regex: str,
    first: bool = True,
    collapse: bool = True,
    strip_d: bool = True,
    omit_nl: bool = False,
    sep: str = ",",
) -> Union[str, List[str]]:
    """Generic segment extractor — port of ``alakazam::getSegment``."""
    scalar = _is_str(segment_call)
    calls = [segment_call] if scalar else list(segment_call)

    edge = f"[^{re.escape(sep)}]*"
    seg_pat = re.compile(edge + "(" + segment_regex + ")" + edge)
    nl_pat = None
    allele_pat = None
    if omit_nl:
        allele_pat = re.compile(edge + "(" + _ALLELE_REGEX + ")" + edge)
        nl_regex = (
            r"(IG[HKL]|TR[ABDG])[VDJADEGMC][0-9]+-NL[0-9]"
            r"([-/\w]*[-\*][\.\w]+)*(" + re.escape(sep) + r"|$)"
        )
        nl_pat = re.compile(nl_regex)
    strip_pat = re.compile(
        r"(?<=[A-Z0-9][0-9])D(?=\*|-|" + re.escape(sep) + r"|$)"
    )

    out: List[str] = []
    for call in calls:
        if call is None or (not _is_str(call)):
            out.append(call if _is_str(call) else "")
            continue
        r = call
        if omit_nl:
            r = allele_pat.sub(lambda m: m.group(1), r)
            r = nl_pat.sub("", r)
        # Replace the first matching substring (gsub replaces all,
        # but the edge-anchored pattern collapses the whole string).
        r = seg_pat.sub(lambda m: m.group(1), r)
        if strip_d:
            r = strip_pat.sub("", r)
        if first:
            r = re.sub(re.escape(sep) + ".*$", "", r)
        elif collapse:
            parts = r.split(sep)
            seen: List[str] = []
            for p in parts:
                if p not in seen:
                    seen.append(p)
            r = sep.join(seen)
        out.append(r)

    return out[0] if scalar else out


def get_gene(segment_call, first=True, collapse=True, strip_d=True,
             omit_nl=False, sep=","):
    """Extract gene name(s) — port of ``alakazam::getGene``."""
    return get_segment(segment_call, _GENE_REGEX, first=first,
                       collapse=collapse, strip_d=strip_d,
                       omit_nl=omit_nl, sep=sep)


def get_allele(segment_call, first=True, collapse=True, strip_d=True,
               omit_nl=False, sep=","):
    """Extract allele name(s) — port of ``alakazam::getAllele``."""
    return get_segment(segment_call, _ALLELE_REGEX, first=first,
                       collapse=collapse, strip_d=strip_d,
                       omit_nl=omit_nl, sep=sep)


def get_family(segment_call, first=True, collapse=True, strip_d=True,
               omit_nl=False, sep=","):
    """Extract family name(s) — port of ``alakazam::getFamily``."""
    return get_segment(segment_call, _FAMILY_REGEX, first=first,
                       collapse=collapse, strip_d=strip_d,
                       omit_nl=omit_nl, sep=sep)


# ---------------------------------------------------------------------------
# DNA translation — port of alakazam::translateDNA (seqinr genetic code,
# ambiguous codons resolved when unambiguous).
# ---------------------------------------------------------------------------
_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}


def _translate_codon(codon: str) -> str:
    codon = codon.upper()
    if codon in _CODON_TABLE:
        return _CODON_TABLE[codon]
    # Resolve ambiguous codons: if every expansion gives the same aa, use it.
    try:
        opts = [_IUPAC[c] for c in codon]
    except KeyError:
        return "X"
    aas = set()
    for a in opts[0]:
        for b in opts[1]:
            for c in opts[2]:
                aas.add(_CODON_TABLE.get(a + b + c, "X"))
    if len(aas) == 1:
        return aas.pop()
    return "X"


def translate_dna(seq: Union[str, Sequence[str]], trim: bool = False):
    """Translate nucleotide sequence(s) to amino acids.

    Port of ``alakazam::translateDNA``.  Gaps (``-``/``.``) are converted to
    ``N`` before translation.  When ``trim`` is ``True`` the first and last
    three nucleotides (conserved codons) are removed first.
    """
    scalar = _is_str(seq)
    seqs = [seq] if scalar else list(seq)
    out: List = []
    for s in seqs:
        if s is None or not _is_str(s):
            out.append(None)
            continue
        x = s
        if trim:
            x = x[3:len(x) - 3] if len(x) > 6 else ""
        x = x.replace("-", "N").replace(".", "N")
        if len(x) >= 3:
            aa = "".join(
                _translate_codon(x[i:i + 3])
                for i in range(0, len(x) - len(x) % 3, 3)
            )
            out.append(aa)
        else:
            out.append(None)
    return out[0] if scalar else out
