"""Smoke tests for pytigger — exercise every public function end-to-end."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pytigger as tg

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures (module-scoped — the trifecta is run once).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def airrdb():
    return tg.load_airrdb()


@pytest.fixture(scope="module")
def germline():
    return tg.load_sample_germline_ighv()


@pytest.fixture(scope="module")
def novel(airrdb, germline):
    return tg.find_novel_alleles(airrdb, germline)


@pytest.fixture(scope="module")
def genotype(airrdb, germline, novel):
    return tg.infer_genotype(airrdb, germline_db=germline, novel=novel,
                             find_unmutated=True)


# ---------------------------------------------------------------------------
# Built-in data
# ---------------------------------------------------------------------------
def test_load_airrdb(airrdb):
    assert isinstance(airrdb, pd.DataFrame)
    assert airrdb.shape[0] == 17559
    for c in ("v_call", "j_call", "sequence_alignment", "junction",
              "junction_length"):
        assert c in airrdb.columns


def test_load_germline(germline):
    assert len(germline) == 344
    assert all(isinstance(v, str) for v in germline.values())
    assert all(name.startswith("IGHV") for name in germline)


def test_load_sample_novel():
    sn = tg.load_sample_novel()
    assert isinstance(sn, pd.DataFrame)
    assert "polymorphism_call" in sn.columns


def test_load_sample_genotype():
    sg = tg.load_sample_genotype()
    assert list(sg.columns)[:4] == ["gene", "alleles", "counts", "total"]
    assert len(sg) == 9


# ---------------------------------------------------------------------------
# Segment parsing
# ---------------------------------------------------------------------------
def test_get_gene_allele_family():
    assert tg.get_gene("IGHV1-69D*01", strip_d=False) == "IGHV1-69D"
    assert tg.get_gene("IGHV1-69D*01", strip_d=True) == "IGHV1-69"
    assert tg.get_allele("Homsap IGHV1-2*01 F", first=False,
                         strip_d=False) == "IGHV1-2*01"
    assert tg.get_family("IGHV1-8*02_G234T") == "IGHV1"
    assert tg.get_gene("IGHV1-8*02_G234T", strip_d=False) == "IGHV1-8"


def test_translate_dna():
    assert tg.translate_dna("GAGGTGCAGCTG") == "EVQL"
    assert tg.translate_dna("TAA") == "*"


# ---------------------------------------------------------------------------
# Sequence / IO utilities
# ---------------------------------------------------------------------------
def test_clean_seqs():
    res = tg.clean_seqs(["AGAT.taa-GAG", "GATXXACA"])
    assert res == ["AGAT.TAA-GAG", "GATNNACA"]


def test_insert_polymorphisms():
    assert tg.insert_polymorphisms("HUGGED", [1, 6, 2],
                                   ["T", "R", "I"]) == "TIGGER"


def test_super_substring():
    assert tg.super_substring("ABCDEFG", [1, 3, 7]) == "ACG"


def test_get_mutated_positions():
    res = tg.get_mutated_positions(["----GATA", "GAGAGAGA", "TANA"],
                                   "GATAGATA")
    assert res == [[], [3, 7], [1]]


def test_update_allele_names():
    assert tg.update_allele_names(["IGHV1-c*01", "IGHV2-5*07"]) == [
        "IGHV1-38-4*01", "IGHV2-5*04"]


def test_sort_alleles():
    al = ["IGHV1-69*02", "IGHV1-2*01", "IGHV1-69*01"]
    assert tg.sort_alleles(al) == ["IGHV1-2*01", "IGHV1-69*01", "IGHV1-69*02"]


def test_write_read_fasta(tmp_path):
    seqs = {"IGHV1-1*01": "ACGTACGT", "IGHV1-2*01": "TTTTGGGG"}
    f = tmp_path / "test.fasta"
    tg.write_fasta(seqs, str(f), width=4)
    back = tg.read_ig_fasta(str(f), strip_down_name=False)
    assert back == seqs


def test_get_mut_count(germline):
    names = list(germline.keys())[:2]
    sub = {n: germline[n] for n in names}
    res = tg.get_mut_count([germline[names[0]]], [names[0]], sub)
    assert res[0] == 0


def test_find_unmutated_calls(germline):
    names = list(germline.keys())[:3]
    sub = {n: germline[n] for n in names}
    calls = tg.find_unmutated_calls(names, [germline[n] for n in names], sub)
    assert set(calls) == set(names)


# ---------------------------------------------------------------------------
# The TIgGER trifecta
# ---------------------------------------------------------------------------
def test_find_novel_alleles(novel):
    assert isinstance(novel, pd.DataFrame)
    assert novel.shape == (12, 30)
    sn = tg.select_novel(novel)
    assert len(sn) == 1
    assert sn.iloc[0]["polymorphism_call"] == "IGHV1-8*02_G234T"
    assert sn.iloc[0]["note"] == "Novel allele found!"
    assert sn.iloc[0]["nt_substitutions"] == "234G>T"


def test_get_popular_mutation_count(airrdb, germline):
    gpm = tg.get_popular_mutation_count(airrdb, germline)
    assert isinstance(gpm, pd.DataFrame)
    assert "mutation_count" in gpm.columns
    assert (gpm["mutation_count"] > 0).all()


def test_infer_genotype(genotype):
    assert list(genotype.columns) == ["gene", "alleles", "counts",
                                      "total", "note"]
    assert len(genotype) == 9
    row = genotype[genotype["gene"] == "IGHV1-8"].iloc[0]
    assert row["alleles"] == "01,02_G234T"


def test_infer_genotype_bayesian(airrdb, germline, novel):
    gb = tg.infer_genotype_bayesian(airrdb, germline_db=germline,
                                    novel=novel, find_unmutated=True)
    for c in ("kh", "kd", "kt", "kq", "k_diff"):
        assert c in gb.columns
    assert len(gb) == 9
    # k_diff is always non-negative (best minus second-best).
    assert (gb["k_diff"] >= 0).all()


def test_genotype_fasta(genotype, germline, novel):
    gdb = tg.genotype_fasta(genotype, germline, novel)
    assert len(gdb) == 15
    assert "IGHV1-8*02_G234T" in gdb


def test_reassign_alleles(airrdb, genotype, germline, novel):
    gdb = tg.genotype_fasta(genotype, germline, novel)
    out = tg.reassign_alleles(airrdb, gdb)
    assert "v_call_genotyped" in out.columns
    assert len(out) == len(airrdb)
    assert (out["v_call_genotyped"].astype(str) != "").all()


def test_generate_evidence(airrdb, genotype, germline, novel):
    gdb = tg.genotype_fasta(genotype, germline, novel)
    out = tg.reassign_alleles(airrdb, gdb)
    ev = tg.generate_evidence(out, novel, genotype, gdb, germline)
    assert len(ev) == 1
    rec = ev.iloc[0]
    assert rec["polymorphism_call"] == "IGHV1-8*02_G234T"
    assert rec["closest_reference"] == "IGHV1-8*02"
    assert rec["nt_diff"] == 1
    assert rec["sequences"] == 864


def test_subsample_db(airrdb):
    # Gene mode: equal number of sequences per gene group, reproducible.
    ss = tg.subsample_db(airrdb, random_state=1)
    assert isinstance(ss, pd.DataFrame)
    assert list(ss.columns) == list(airrdb.columns)
    assert 0 < len(ss) <= len(airrdb)
    # Reproducible for a fixed seed.
    ss_again = tg.subsample_db(airrdb, random_state=1)
    assert ss.index.tolist() == ss_again.index.tolist()
    # Allele mode with an explicit cap.
    ss_allele = tg.subsample_db(airrdb, mode="allele", max_n=10,
                                random_state=1)
    assert 0 < len(ss_allele) <= len(airrdb)
    # Additional grouping variable is subsampled independently.
    tagged = airrdb.copy()
    tagged["sample_id"] = (["A", "B"] * (len(tagged) // 2 + 1))[:len(tagged)]
    ss_grp = tg.subsample_db(tagged, group="sample_id", random_state=1)
    assert list(ss_grp.columns) == list(tagged.columns)
    assert 0 < len(ss_grp) <= len(tagged)
    # A NumPy Generator is accepted too.
    ss_np = tg.subsample_db(airrdb, random_state=np.random.default_rng(0))
    assert 0 < len(ss_np) <= len(airrdb)
    # Missing / invalid arguments raise informative errors.
    with pytest.raises(ValueError):
        tg.subsample_db(airrdb, gene="not_a_column")
    with pytest.raises(ValueError):
        tg.subsample_db(airrdb, mode="bogus")


def test_evidence_helpers():
    assert tg.has_non_imgt_gaps("AC-GT") is True
    assert tg.has_non_imgt_gaps("ACGTAC") is False
    muts = tg.get_mutated_aa("GAGGTGCAGCTG", "GAGGTGCAGTTG")
    assert isinstance(muts, list)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def test_plot_novel(airrdb, novel):
    import matplotlib
    matplotlib.use("Agg")
    sn = tg.select_novel(novel)
    fig = tg.plot_novel(airrdb, sn.iloc[[0]], multiplot=True)
    assert fig is not None
    figs = tg.plot_novel(airrdb, sn.iloc[[0]], multiplot=False)
    assert len(figs) == 3


def test_plot_genotype():
    import matplotlib
    matplotlib.use("Agg")
    fig = tg.plot_genotype(tg.load_sample_genotype(), silent=True)
    assert fig is not None
    fig2 = tg.plot_genotype(tg.load_sample_genotype(),
                            gene_sort="position", silent=True)
    assert fig2 is not None
