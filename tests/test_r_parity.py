"""R-parity tests — pytigger vs R/CRAN tigger 1.1.3.

The R driver (:file:`r_reference_driver.R`) runs the TIgGER trifecta on
tigger's own bundled example data (``AIRRDb`` + ``SampleGermlineIGHV``), so
both sides analyse the exact same input.  We compare:

* ``find_novel_alleles``      — every output column matches R exactly
  (deterministic algorithm).
* ``infer_genotype``          — gene / allele / count membership and the
  notes match R exactly.
* ``infer_genotype_bayesian`` — discrete genotype matches R; the log10
  likelihoods agree to rel-diff < 1e-6.
* ``genotype_fasta``          — same allele set and sequences as R.
* ``reassign_alleles``        — the v_call_genotyped column matches R on
  100% of sequences.

Tests skip gracefully when the CMAP R environment or tigger is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pytigger as tg

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_R_OUT = _HERE / "r_ref_out"

# CMAP R environment (per project conventions).
_R_SETUP = (
    "source /home/users/steorra/miniforge3/etc/profile.d/conda.sh && "
    "conda activate /scratch/users/steorra/env/CMAP && "
)


def _r_available() -> bool:
    """Return True if R + tigger are importable in the CMAP env."""
    try:
        res = subprocess.run(
            _R_SETUP + "Rscript -e 'library(tigger)'",
            shell=True, capture_output=True, text=True, timeout=120,
            executable="/bin/bash",
        )
        return res.returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def r_outputs():
    """Run the R reference driver once; return the output directory."""
    if not _r_available():
        pytest.skip("R / tigger not available in CMAP environment")
    _R_OUT.mkdir(exist_ok=True)
    cmd = _R_SETUP + (
        f"Rscript {_HERE / 'r_reference_driver.R'} {_R_OUT}"
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                         timeout=1800, executable="/bin/bash")
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return _R_OUT


@pytest.fixture(scope="module")
def py_pipeline():
    """Run the Python trifecta once on the bundled example data."""
    data = tg.load_airrdb()
    germ = tg.load_sample_germline_ighv()
    novel = tg.find_novel_alleles(data, germ)
    geno = tg.infer_genotype(data, germline_db=germ, novel=novel,
                             find_unmutated=True)
    geno_b = tg.infer_genotype_bayesian(data, germline_db=germ,
                                        novel=novel, find_unmutated=True)
    gdb = tg.genotype_fasta(geno, germ, novel)
    out = tg.reassign_alleles(data, gdb)
    return {"data": data, "germ": germ, "novel": novel, "geno": geno,
            "geno_b": geno_b, "gdb": gdb, "out": out}


# ---------------------------------------------------------------------------
def test_parity_find_novel_alleles(r_outputs, py_pipeline):
    """Every column of the novel-allele table matches R exactly."""
    r_novel = pd.read_csv(r_outputs / "R_novel.csv")
    py_novel = py_pipeline["novel"]
    assert py_novel.shape == r_novel.shape

    str_cols = ["germline_call", "note", "polymorphism_call",
                "nt_substitutions", "novel_imgt", "germline_imgt"]
    for c in str_cols:
        py = py_novel[c].fillna("NA").astype(str).tolist()
        rr = r_novel[c].fillna("NA").astype(str).tolist()
        assert py == rr, f"column {c} differs from R"

    num_cols = ["novel_imgt_count", "perfect_match_count",
                "germline_call_count", "mut_min", "mut_max",
                "mut_pass_count", "germline_imgt_count", "y_intercept_pass",
                "snp_pass", "unmutated_count",
                "unmutated_snp_j_gene_length_count",
                "snp_min_seqs_j_max_pass"]
    for c in num_cols:
        py = py_novel[c].astype(float).values
        rr = r_novel[c].astype(float).values
        assert np.allclose(np.nan_to_num(py), np.nan_to_num(rr),
                           rtol=1e-9, equal_nan=True), \
            f"column {c} differs from R"


def test_parity_select_novel(r_outputs, py_pipeline):
    """The single detected novel allele matches R."""
    sn = tg.select_novel(py_pipeline["novel"])
    assert len(sn) == 1
    assert sn.iloc[0]["polymorphism_call"] == "IGHV1-8*02_G234T"


def test_parity_infer_genotype(r_outputs, py_pipeline):
    """inferGenotype gene/allele/count/note membership matches R exactly."""
    r_geno = pd.read_csv(r_outputs / "R_genotype.csv",
                         dtype={"alleles": str, "counts": str}).fillna("")
    py_geno = py_pipeline["geno"].fillna("")
    cols = ["gene", "alleles", "counts", "total", "note"]
    py = py_geno[cols].astype(str).reset_index(drop=True)
    rr = r_geno[cols].astype(str).reset_index(drop=True)
    pd.testing.assert_frame_equal(py, rr)


def test_parity_infer_genotype_bayesian(r_outputs, py_pipeline):
    """Bayesian discrete genotype matches; likelihoods rel-diff < 1e-6."""
    r_gb = pd.read_csv(r_outputs / "R_genotype_bayes.csv",
                       dtype={"alleles": str, "counts": str}).fillna("")
    py_gb = py_pipeline["geno_b"].fillna("")

    # Discrete part.
    cols = ["gene", "alleles", "counts", "total", "note"]
    py = py_gb[cols].astype(str).reset_index(drop=True)
    rr = r_gb[cols].astype(str).reset_index(drop=True)
    pd.testing.assert_frame_equal(py, rr)

    # Continuous likelihoods.
    for c in ["kh", "kd", "kt", "kq", "k_diff"]:
        py_v = py_gb[c].astype(float).values
        rr_v = r_gb[c].astype(float).values
        rel = np.abs(py_v - rr_v) / (np.abs(rr_v) + 1e-12)
        assert rel.max() < 1e-6, f"{c}: max rel-diff {rel.max():.2e}"


def test_parity_genotype_fasta(r_outputs, py_pipeline):
    """genotypeFasta returns the same alleles and sequences as R."""
    r_gtdb = pd.read_csv(r_outputs / "R_gtdb.csv")
    r_map = dict(zip(r_gtdb["name"], r_gtdb["seq"]))
    py_gdb = py_pipeline["gdb"]
    assert set(py_gdb.keys()) == set(r_map.keys())
    for name, seq in r_map.items():
        assert py_gdb[name] == seq, f"sequence for {name} differs"


def test_parity_reassign_alleles(r_outputs, py_pipeline):
    """reassignAlleles v_call_genotyped matches R on 100% of sequences."""
    r_reassign = pd.read_csv(r_outputs / "R_reassign.csv")
    py_calls = py_pipeline["out"]["v_call_genotyped"].astype(str).values
    r_calls = r_reassign["v_call_genotyped"].astype(str).values
    assert len(py_calls) == len(r_calls)
    agree = (py_calls == r_calls).mean()
    assert agree == 1.0, f"reassignAlleles agreement {agree:.4%}"


def test_parity_generate_evidence(r_outputs, py_pipeline):
    """generateEvidence metrics match R."""
    r_ev = pd.read_csv(r_outputs / "R_evidence.csv")
    p = py_pipeline
    ev = tg.generate_evidence(p["out"], p["novel"], p["geno"], p["gdb"],
                              p["germ"])
    assert len(ev) == len(r_ev) == 1
    py_rec, r_rec = ev.iloc[0], r_ev.iloc[0]
    assert py_rec["polymorphism_call"] == r_rec["polymorphism_call"]
    assert py_rec["closest_reference"] == r_rec["closest_reference"]
    for c in ["nt_diff", "aa_diff", "sequences", "unmutated_sequences",
              "unique_js", "unique_cdr3s"]:
        assert float(py_rec[c]) == float(r_rec[c]), f"{c} differs"
    assert abs(float(py_rec["allelic_percentage"])
               - float(r_rec["allelic_percentage"])) < 1e-6
