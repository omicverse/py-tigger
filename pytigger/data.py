"""Built-in example data — the datasets bundled with the R tigger package.

* :func:`load_airrdb`              — example human AIRR-seq repertoire
  (``AIRRDb`` in R; 17,559 sequences).
* :func:`load_sample_germline_ighv` — example human IGHV germline reference
  (``SampleGermlineIGHV``; 344 alleles).
* :func:`load_sample_novel`        — example ``findNovelAlleles`` output
  (``SampleNovel``).
* :func:`load_sample_genotype`     — example ``inferGenotype`` output
  (``SampleGenotype``).
"""
from __future__ import annotations

import gzip
import io
import os
from collections import OrderedDict
from typing import Dict

import pandas as pd

from .sequences import read_ig_fasta

__all__ = [
    "load_airrdb",
    "load_sample_germline_ighv",
    "load_sample_novel",
    "load_sample_genotype",
]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_airrdb() -> pd.DataFrame:
    """Load the bundled ``AIRRDb`` example repertoire data frame."""
    path = os.path.join(_DATA_DIR, "AIRRDb.csv.gz")
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh)
    return df


def load_sample_germline_ighv() -> "OrderedDict[str, str]":
    """Load the bundled ``SampleGermlineIGHV`` germline reference."""
    path = os.path.join(_DATA_DIR, "SampleGermlineIGHV.fasta")
    seqs = read_ig_fasta(path, strip_down_name=False, force_caps=True)
    return OrderedDict(seqs)


def load_sample_novel() -> pd.DataFrame:
    """Load the bundled ``SampleNovel`` (findNovelAlleles output)."""
    path = os.path.join(_DATA_DIR, "SampleNovel.csv")
    return pd.read_csv(path)


def load_sample_genotype() -> pd.DataFrame:
    """Load the bundled ``SampleGenotype`` (inferGenotype output)."""
    path = os.path.join(_DATA_DIR, "SampleGenotype.csv")
    df = pd.read_csv(path, dtype={"alleles": str, "counts": str})
    if "note" in df.columns:
        df["note"] = df["note"].fillna("")
    return df
