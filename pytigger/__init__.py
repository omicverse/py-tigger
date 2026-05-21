"""pytigger: Pure-Python port of the R/CRAN package **tigger**.

``tigger`` (Tools for Immunoglobulin Genotype Elucidation via Rep-seq) is
part of the `Immcantation <https://immcantation.readthedocs.io>`_ framework
(Kleinstein lab, Yale).  It discovers novel immunoglobulin V alleles from
adaptive immune receptor repertoire sequencing data (AIRR-Seq / Rep-Seq),
infers a subject's V genotype, and corrects V-allele calls accordingly.

This is a faithful, dependency-light Python re-implementation
(numpy / scipy / pandas / matplotlib, no rpy2) of tigger 1.1.3, with
numerical parity to the R package as the top priority.

The TIgGER trifecta
-------------------
* :func:`find_novel_alleles`     — novel V-allele discovery via the
  mutation-accumulation / y-intercept regression algorithm.
* :func:`infer_genotype`         — frequency-method genotype inference.
* :func:`infer_genotype_bayesian` — Bayesian Dirichlet-multinomial genotype
  inference.
* :func:`reassign_alleles`       — correct V calls using the genotype.

Supporting functions
--------------------
* :func:`select_novel`, :func:`genotype_fasta`, :func:`generate_evidence`,
  :func:`get_popular_mutation_count`.
* Sequence/IO: :func:`read_ig_fasta`, :func:`write_fasta`,
  :func:`clean_seqs`, :func:`update_allele_names`, :func:`sort_alleles`,
  :func:`get_mutated_positions`, :func:`get_mut_count`,
  :func:`find_unmutated_calls`, :func:`insert_polymorphisms`.
* Segment parsing (alakazam port): :func:`get_gene`, :func:`get_allele`,
  :func:`get_family`, :func:`translate_dna`.
* Plotting: :func:`plot_novel`, :func:`plot_genotype`.

Built-in example data (the datasets bundled with R tigger)
----------------------------------------------------------
* :func:`load_airrdb`, :func:`load_sample_germline_ighv`,
  :func:`load_sample_novel`, :func:`load_sample_genotype`.

Quick-start
-----------
>>> import pytigger as tg
>>> data = tg.load_airrdb()
>>> germ = tg.load_sample_germline_ighv()
>>> novel = tg.find_novel_alleles(data, germ)
>>> tg.select_novel(novel)[["germline_call", "polymorphism_call"]]
>>> geno = tg.infer_genotype(data, germline_db=germ, novel=novel)
>>> gdb = tg.genotype_fasta(geno, germ, novel)
>>> out = tg.reassign_alleles(data, gdb)
"""
from __future__ import annotations

from .data import (
    load_airrdb,
    load_sample_genotype,
    load_sample_germline_ighv,
    load_sample_novel,
)
from .evidence import generate_evidence, get_mutated_aa, has_non_imgt_gaps
from .genotype import (
    genotype_fasta,
    infer_genotype,
    infer_genotype_bayesian,
    reassign_alleles,
)
from .novel import (
    find_novel_alleles,
    get_popular_mutation_count,
    select_novel,
)
from .plotting import plot_genotype, plot_novel
from .segments import get_allele, get_family, get_gene, translate_dna
from .sequences import (
    clean_seqs,
    find_unmutated_calls,
    get_mut_count,
    get_mutated_positions,
    insert_polymorphisms,
    read_ig_fasta,
    sort_alleles,
    super_substring,
    update_allele_names,
    write_fasta,
)

__version__ = "0.1.0"

__all__ = [
    # The TIgGER trifecta + core
    "find_novel_alleles",
    "select_novel",
    "infer_genotype",
    "infer_genotype_bayesian",
    "reassign_alleles",
    "genotype_fasta",
    "generate_evidence",
    "get_popular_mutation_count",
    # sequence / IO utilities
    "read_ig_fasta",
    "write_fasta",
    "clean_seqs",
    "update_allele_names",
    "sort_alleles",
    "get_mutated_positions",
    "get_mut_count",
    "find_unmutated_calls",
    "insert_polymorphisms",
    "super_substring",
    # segment parsing
    "get_gene",
    "get_allele",
    "get_family",
    "translate_dna",
    # evidence helpers
    "has_non_imgt_gaps",
    "get_mutated_aa",
    # plotting
    "plot_novel",
    "plot_genotype",
    # built-in data
    "load_airrdb",
    "load_sample_germline_ighv",
    "load_sample_novel",
    "load_sample_genotype",
]
