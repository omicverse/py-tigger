"""Repertoire subsampling — faithful port of tigger's ``subsampleDb``.

``subsampleDb`` subsamples an AIRR/Rep-seq database so that an equal number
of sequences is drawn from every gene, allele or family group.  It is used
to balance a repertoire before novel-allele discovery or genotype
inference, so that highly-expressed genes do not dominate the analysis.

To reproduce R's results bit-for-bit, this module also ships a faithful
re-implementation of R's Mersenne-Twister RNG and of ``sample.int`` (with
the post-R-3.6 "Rejection" index method).  When ``subsample_db`` is given an
integer seed it draws through this R-compatible stream, so the selected rows
match ``tigger::subsampleDb`` exactly after the same ``set.seed``.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .segments import get_allele, get_family, get_gene

__all__ = ["subsample_db"]


# ---------------------------------------------------------------------------
# R-compatible Mersenne-Twister RNG (matches set.seed + sample.int exactly).
# ---------------------------------------------------------------------------
class _RRNG:
    """R's Mersenne-Twister, bit-exact with ``set.seed`` and ``sample.int``.

    Reproduces R's RNG initial scrambling (``69069 * seed + 1``), the
    MT19937 generator, ``unif_rand``'s ``fixup`` clamping, and the
    post-R-3.6 ``R_unif_index`` "Rejection" sampling used by ``sample``.
    """

    _N = 624
    _M = 397
    _MATRIX_A = 0x9908B0DF
    _UPPER = 0x80000000
    _LOWER = 0x7FFFFFFF

    def __init__(self, seed: int):
        s = int(seed) & 0xFFFFFFFF
        # R's RNG_Init initial scrambling.
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        # Fill the 625 i_seed slots; i_seed[0] is mti, i_seed[1:] the state.
        i_seed: List[int] = []
        for _ in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            i_seed.append(s)
        self._mt = i_seed[1:]
        self._mti = self._N  # FixupSeeds forces a regeneration on first draw.

    def _genrand(self) -> int:
        mag01 = (0, self._MATRIX_A)
        mt = self._mt
        if self._mti >= self._N:
            for kk in range(self._N - self._M):
                y = (mt[kk] & self._UPPER) | (mt[kk + 1] & self._LOWER)
                mt[kk] = mt[kk + self._M] ^ (y >> 1) ^ mag01[y & 1]
            for kk in range(self._N - self._M, self._N - 1):
                y = (mt[kk] & self._UPPER) | (mt[kk + 1] & self._LOWER)
                mt[kk] = (mt[kk + (self._M - self._N)]
                          ^ (y >> 1) ^ mag01[y & 1])
            y = (mt[self._N - 1] & self._UPPER) | (mt[0] & self._LOWER)
            mt[self._N - 1] = mt[self._M - 1] ^ (y >> 1) ^ mag01[y & 1]
            self._mti = 0
        y = mt[self._mti]
        self._mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def unif_rand(self) -> float:
        """One draw from ``runif(1)`` — R's ``unif_rand`` with ``fixup``."""
        v = self._genrand() * 2.3283064365386963e-10
        if v <= 0.0:
            return 0.5 * 2.328306437080797e-10
        if v >= 1.0:
            return 1.0 - 0.5 * 2.328306437080797e-10
        return v

    def _rbits(self, bits: int) -> int:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """R's ``R_unif_index`` — Rejection method (R >= 3.6)."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dv < dn:
                return dv

    def sample_int(self, n: int, k: int) -> List[int]:
        """R's ``sample.int(n, k, replace = FALSE)`` — returns 0-based picks."""
        ix = list(range(n))
        out: List[int] = []
        nn = n
        for _ in range(k):
            j = self.unif_index(nn)
            out.append(ix[j])
            ix[j] = ix[nn - 1]
            nn -= 1
        return out


def _check_columns(data: pd.DataFrame,
                    columns: Sequence[str]) -> Union[bool, str]:
    """Port of ``alakazam::checkColumns`` (``logic = "all"``).

    Returns ``True`` when every requested column is present and contains at
    least one non-NA value; otherwise returns an explanatory message.
    """
    data_names = list(data.columns)
    for f in columns:
        if f is None:
            continue
        if f not in data_names:
            return f"The column {f} was not found"
    for f in columns:
        if f is None:
            continue
        if data[f].isna().all():
            return f"The column {f} contains no data"
    return True


def subsample_db(
    data: pd.DataFrame,
    gene: str = "v_call",
    mode: str = "gene",
    min_n: int = 1,
    max_n: Optional[int] = None,
    group: Optional[Union[str, Sequence[str]]] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> pd.DataFrame:
    """Subsample a repertoire to an equal number of sequences per group.

    Faithful port of ``tigger::subsampleDb``.  ``data`` is split into gene,
    allele or family subsets (``mode``) from which the same number of
    sequences is subsampled.  Sequences with multiple gene calls (ties) may
    be subsampled from any of their calls, but duplicated samplings are
    removed and the returned frame contains unique rows.

    Parameters
    ----------
    data : pandas.DataFrame
        ``data.frame`` containing repertoire data.
    gene : str, default ``"v_call"``
        Name of the column in ``data`` with allele calls.
    mode : {"gene", "allele", "family"}, default ``"gene"``
        Degree of specificity used when splitting ``data`` into the subsets
        from which the same number of sequences will be subsampled.
    min_n : int, default 1
        Minimum number of observations a group must have to be sampled.  A
        group with fewer observations than ``min_n`` is excluded.
    max_n : int, optional
        Maximum number of observations to sample for all ``mode`` groups.
        If ``None``, it is set automatically to the size of the smallest
        group.  If ``max_n`` is larger than the available number of
        sequences for any group, the effective ``max_n`` used is the size
        of the smallest ``mode`` group.
    group : str or sequence of str, optional
        Column(s) containing additional grouping variables, e.g.
        ``sample_id``.  These groups are subsampled independently.  When
        ``max_n`` is ``None`` a ``max_n`` is set automatically for each
        group.
    random_state : int or numpy.random.Generator, optional
        Controls the random sampling so results are reproducible.  An
        *integer* seed draws through an R-compatible Mersenne-Twister
        stream, so the selected rows match ``tigger::subsampleDb`` exactly
        after the same ``set.seed``.  A :class:`numpy.random.Generator`
        uses NumPy's RNG instead.  ``None`` is non-deterministic.

    Returns
    -------
    pandas.DataFrame
        A subsampled version of the input ``data`` (a subset of its rows,
        with the original column set and order preserved).

    See Also
    --------
    select_novel

    Examples
    --------
    >>> import pytigger as tg
    >>> data = tg.load_airrdb()
    >>> ss = tg.subsample_db(data, random_state=1)
    >>> len(ss) <= len(data)
    True
    """
    mode = str(mode).lower()
    if mode not in ("gene", "allele", "family"):
        raise ValueError("mode must be one of 'gene', 'allele', 'family'")

    # --- additional grouping variables -------------------------------------
    if group is not None:
        group_cols = [group] if isinstance(group, str) else list(group)
    else:
        group_cols = []

    check = _check_columns(data, [gene] + group_cols)
    if check is not True:
        raise ValueError(check)

    # Choose the RNG: an integer seed -> R-compatible stream (bit-exact with
    # tigger); a numpy Generator -> numpy stream; otherwise non-deterministic.
    if isinstance(random_state, (_RRNG, np.random.Generator)):
        rng = random_state
    elif random_state is None:
        rng = np.random.default_rng()
    else:
        rng = _RRNG(int(random_state))

    def _sample(idx: Sequence[int], n: int) -> List[int]:
        """Sample ``n`` items from ``idx`` without replacement."""
        arr = list(idx)
        if isinstance(rng, _RRNG):
            picks = rng.sample_int(len(arr), n)
            return [arr[p] for p in picks]
        return [int(p) for p in rng.choice(np.asarray(arr, dtype=np.int64),
                                           size=n, replace=False)]

    # Recurse over additional grouping variables (e.g. sample_id): each
    # group is subsampled independently and the results are concatenated.
    if group_cols:
        keys = (data[group_cols[0]] if len(group_cols) == 1
                else list(zip(*[data[c] for c in group_cols])))
        # dplyr's group_indices() orders groups by the sorted group key;
        # split() then iterates the groups in that order.
        sub = pd.Series(list(keys), index=data.index)
        pieces: List[pd.DataFrame] = []
        for _, idx in sorted(sub.groupby(sub, sort=True).groups.items()):
            piece = data.loc[idx]
            pieces.append(
                subsample_db(piece, gene=gene, mode=mode, min_n=min_n,
                             max_n=max_n, group=None, random_state=rng)
            )
        if not pieces:
            return data.iloc[0:0].copy()
        return pd.concat(pieces, axis=0)

    # --- single-group subsampling ------------------------------------------
    gene_func = {"allele": get_allele, "gene": get_gene,
                 "family": get_family}[mode]
    ss_gene = gene_func(list(data[gene]), first=False)

    # Unique genes across all (possibly comma-separated) calls, first-seen.
    genes: List[str] = []
    seen = set()
    for call in ss_gene:
        for g in str(call).split(","):
            if g and g not in seen:
                seen.add(g)
                genes.append(g)

    # For each gene, the positional indices of every row whose call string
    # contains that gene (fixed substring match, mirroring R's grep()).
    n_rows = len(ss_gene)
    allele_groups = {
        g: [i for i in range(n_rows) if g in str(ss_gene[i])]
        for g in genes
    }

    group_sizes = {g: len(idx) for g, idx in allele_groups.items()}
    # Drop groups smaller than min_n.
    kept = {g: idx for g, idx in allele_groups.items()
            if group_sizes[g] >= min_n}
    if not kept:
        raise ValueError(
            "Not enough sample sequences (min_n) were assigned to any "
            "gene. Returned `NULL`"
        )

    sizes = list(group_sizes.values())
    if max_n is not None:
        n = min(min(sizes), max_n)
    else:
        n = min(sizes)

    # Subsample n positional indices from each kept group (without
    # replacement), then take the union of selected rows (unique).
    ss_idx: List[int] = []
    chosen = set()
    for g, idx in kept.items():
        for p in _sample(idx, n):
            if p not in chosen:
                chosen.add(p)
                ss_idx.append(p)

    return data.iloc[ss_idx].copy()
