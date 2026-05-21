"""Benchmark pytigger against the bundled tigger example data.

Times each stage of the TIgGER trifecta and reports the discovered novel
allele, the inferred genotype and the reassignment summary.

Run::

    python examples/benchmark.py
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")

import pytigger as tg


def _timed(label, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<28s} {dt:8.2f} s")
    return result


def main() -> None:
    print("=" * 64)
    print("pytigger benchmark — TIgGER trifecta on AIRRDb / SampleGermlineIGHV")
    print("=" * 64)

    data = tg.load_airrdb()
    germ = tg.load_sample_germline_ighv()
    print(f"\nInput: {len(data):,} sequences, {len(germ)} germline alleles\n")

    print("Timings")
    print("-" * 64)
    novel = _timed("find_novel_alleles", lambda: tg.find_novel_alleles(
        data, germ))
    geno = _timed("infer_genotype", lambda: tg.infer_genotype(
        data, germline_db=germ, novel=novel, find_unmutated=True))
    geno_b = _timed("infer_genotype_bayesian",
                    lambda: tg.infer_genotype_bayesian(
                        data, germline_db=germ, novel=novel,
                        find_unmutated=True))
    gdb = _timed("genotype_fasta", lambda: tg.genotype_fasta(
        geno, germ, novel))
    out = _timed("reassign_alleles", lambda: tg.reassign_alleles(data, gdb))
    ev = _timed("generate_evidence", lambda: tg.generate_evidence(
        out, novel, geno, gdb, germ))

    print("\nNovel alleles")
    print("-" * 64)
    sn = tg.select_novel(novel)
    print(sn[["germline_call", "polymorphism_call", "nt_substitutions",
              "note"]].to_string(index=False))

    print("\nInferred genotype (frequency method)")
    print("-" * 64)
    print(geno[["gene", "alleles", "counts", "total"]].to_string(index=False))

    print("\nBayesian genotype — Bayes factor (k_diff) per gene")
    print("-" * 64)
    print(geno_b[["gene", "alleles", "k_diff"]].to_string(index=False))

    print("\nAllele reassignment")
    print("-" * 64)
    changed = (out["v_call_genotyped"].astype(str)
               != out["v_call"].astype(str)).sum()
    print(f"  genotype germlines : {len(gdb)}")
    print(f"  sequences corrected: {changed:,} / {len(out):,}")

    print("\nEvidence table")
    print("-" * 64)
    print(ev[["polymorphism_call", "closest_reference", "nt_diff",
              "sequences", "unmutated_sequences",
              "allelic_percentage"]].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
