# pytigger

**Pure-Python port of the R/CRAN package [`tigger`](https://tigger.readthedocs.io)**
— *Tools for Immunoglobulin Genotype Elucidation via Rep-seq.*

`tigger` is part of the [Immcantation](https://immcantation.readthedocs.io)
framework (Kleinstein Lab, Yale). It discovers **novel immunoglobulin V
alleles** from adaptive immune receptor repertoire sequencing data
(AIRR-Seq / Rep-Seq), infers a subject's **V genotype**, and **corrects**
V-allele calls accordingly.

`pytigger` is a faithful, dependency-light re-implementation of tigger
**1.1.3** in pure Python (`numpy` / `scipy` / `pandas` / `matplotlib`,
**no rpy2**). Numerical parity with the R package is the design priority:
on the bundled example data the novel-allele table, genotype membership and
reassigned calls **match R exactly**, and the Bayesian likelihood numbers
agree to a relative difference below `1e-13`.

## Installation

```bash
pip install pytigger
```

From source:

```bash
git clone https://github.com/omicverse/py-tigger
cd py-tigger
pip install -e .
```

## The TIgGER trifecta

```python
import pytigger as tg

# Bundled example data (the datasets shipped with R tigger)
data = tg.load_airrdb()                 # 17,559 AIRR-seq sequences
germ = tg.load_sample_germline_ighv()   # 344 IGHV germline alleles

# 1. Discover novel V alleles
novel = tg.find_novel_alleles(data, germ)
tg.select_novel(novel)[["germline_call", "polymorphism_call", "note"]]
#   germline_call  polymorphism_call               note
#   IGHV1-8*02     IGHV1-8*02_G234T   Novel allele found!

# 2. Infer the subject's V genotype
geno = tg.infer_genotype(data, germline_db=germ, novel=novel,
                         find_unmutated=True)
# ... or the full Bayesian model
geno_b = tg.infer_genotype_bayesian(data, germline_db=germ, novel=novel)

# 3. Correct the V-call assignments
gdb = tg.genotype_fasta(geno, germ, novel)
out = tg.reassign_alleles(data, gdb)        # adds 'v_call_genotyped'

# Evidence table for the inferred novel alleles
ev = tg.generate_evidence(out, novel, geno, gdb, germ)
```

## Visualisation

```python
fig1 = tg.plot_novel(data, tg.select_novel(novel).iloc[[0]])  # evidence plot
fig2 = tg.plot_genotype(tg.load_sample_genotype())            # genotype grid
```

## API

### The trifecta + core

| function | R equivalent |
|---|---|
| `find_novel_alleles` | `findNovelAlleles` |
| `select_novel` | `selectNovel` |
| `infer_genotype` | `inferGenotype` |
| `infer_genotype_bayesian` | `inferGenotypeBayesian` |
| `reassign_alleles` | `reassignAlleles` |
| `genotype_fasta` | `genotypeFasta` |
| `generate_evidence` | `generateEvidence` |
| `get_popular_mutation_count` | `getPopularMutationCount` |

### Sequence / IO utilities

`read_ig_fasta`, `write_fasta`, `clean_seqs`, `update_allele_names`,
`sort_alleles`, `get_mutated_positions`, `get_mut_count`,
`find_unmutated_calls`, `insert_polymorphisms`, `super_substring`.

### Segment parsing (alakazam port)

`get_gene`, `get_allele`, `get_family`, `translate_dna`.

### Plotting

`plot_novel`, `plot_genotype`.

### Built-in data

`load_airrdb`, `load_sample_germline_ighv`, `load_sample_novel`,
`load_sample_genotype`.

## R-parity

`pytigger` is validated against `tigger` 1.1.3 (installed alongside
`alakazam` 1.4.3). Running the trifecta on the bundled `AIRRDb` /
`SampleGermlineIGHV` example data:

* **`find_novel_alleles`** — all 30 output columns match R for all 12 rows;
  the single novel allele `IGHV1-8*02_G234T` is detected identically.
* **`infer_genotype`** — gene/allele/count membership and the
  "Cannot distinguish" notes match R exactly.
* **`infer_genotype_bayesian`** — discrete genotype matches R; the `kh`,
  `kd`, `kt`, `kq`, `k_diff` log-likelihoods agree to rel-diff `< 1e-13`.
* **`reassign_alleles`** — the `v_call_genotyped` column matches R on
  100% of 17,559 sequences.
* **`generate_evidence`** — the evidence metrics match R exactly.

See `tests/test_r_parity.py` (skipped automatically when no R is available).

## Citation

If you use `pytigger`, please cite the original tigger papers:

> Gadala-Maria D, Yaari G, Uduman M, Kleinstein SH (2015).
> *Automated analysis of high-throughput B-cell sequencing data reveals a
> high frequency of novel immunoglobulin V gene segment alleles.*
> PNAS 112(8):E862-70.

> Gadala-Maria D, Gidoni M, Marquez S, et al. (2019).
> *Identification of Subject-Specific Immunoglobulin Alleles from Expressed
> Repertoire Sequencing Data.* Front. Immunol. 10:129.

## License

AGPL-3, retained from the original tigger package. See `LICENSE`.
