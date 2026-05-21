#!/usr/bin/env Rscript
# R reference driver for py-tigger R-parity tests.
#
# Runs the TIgGER trifecta (findNovelAlleles, inferGenotype,
# inferGenotypeBayesian, reassignAlleles) on tigger's own bundled example
# data (AIRRDb + SampleGermlineIGHV) and writes the results as CSV so the
# Python test suite can compare against them.
#
# Usage:  Rscript r_reference_driver.R <output_dir>

suppressMessages(library(tigger))

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "r_ref_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data(AIRRDb)
data(SampleGermlineIGHV)

# --- findNovelAlleles -------------------------------------------------------
novel <- findNovelAlleles(AIRRDb, SampleGermlineIGHV, nproc = 1)
write.csv(novel, file.path(out_dir, "R_novel.csv"), row.names = FALSE)

# --- inferGenotype ----------------------------------------------------------
geno <- inferGenotype(AIRRDb, germline_db = SampleGermlineIGHV,
                      novel = novel, find_unmutated = TRUE)
write.csv(geno, file.path(out_dir, "R_genotype.csv"), row.names = FALSE)

# --- inferGenotypeBayesian --------------------------------------------------
geno_b <- inferGenotypeBayesian(AIRRDb, germline_db = SampleGermlineIGHV,
                                novel = novel, find_unmutated = TRUE)
write.csv(geno_b, file.path(out_dir, "R_genotype_bayes.csv"),
          row.names = FALSE)

# --- genotypeFasta + reassignAlleles ---------------------------------------
gtdb <- genotypeFasta(geno, SampleGermlineIGHV, novel)
write.csv(data.frame(name = names(gtdb), seq = as.character(gtdb)),
          file.path(out_dir, "R_gtdb.csv"), row.names = FALSE)

out <- reassignAlleles(AIRRDb, gtdb, v_call = "v_call",
                       seq = "sequence_alignment")
write.csv(data.frame(v_call_genotyped = out$v_call_genotyped),
          file.path(out_dir, "R_reassign.csv"), row.names = FALSE)

# --- generateEvidence -------------------------------------------------------
ev <- generateEvidence(out, novel, geno, gtdb, SampleGermlineIGHV,
                       j_call = "j_call", junction = "junction")
write.csv(as.data.frame(ev), file.path(out_dir, "R_evidence.csv"),
          row.names = FALSE)

# --- subsampleDb ------------------------------------------------------------
# Default gene-mode subsampling with a fixed seed.  An explicit `.orig_row`
# column is added so the exact set of selected rows can be recovered (R's
# select() inside subsampleDb resets the data.frame rownames).  We write the
# selected original-row indices and the per-(algorithm-group) sampled counts
# so the Python test can compare RNG-independent invariants (total size and
# per-group counts).
AIRRDb_tagged <- AIRRDb
AIRRDb_tagged[[".orig_row"]] <- seq_len(nrow(AIRRDb))

set.seed(1)
ss_gene <- subsampleDb(AIRRDb_tagged)
ss_idx  <- sort(ss_gene[[".orig_row"]])
write.csv(data.frame(orig_row = ss_idx),
          file.path(out_dir, "R_subsample_gene.csv"), row.names = FALSE)

# Per-(algorithm-group) sampled counts: for every gene, how many sampled
# rows fall in that gene's substring-grep group (mirrors subsampleDb's own
# allele_groups definition).
gf      <- alakazam::getGene(AIRRDb$v_call, first = FALSE)
genes   <- unique(unlist(strsplit(gf, ",")))
groups  <- sapply(genes, grep, gf, fixed = TRUE, simplify = FALSE)
n_gene  <- min(sapply(groups, length))
per_grp <- sapply(groups, function(idx) length(intersect(idx, ss_idx)))
write.csv(data.frame(gene = names(per_grp), count = as.integer(per_grp),
                     n = n_gene, total = nrow(ss_gene)),
          file.path(out_dir, "R_subsample_counts.csv"), row.names = FALSE)

cat("R reference outputs written to", out_dir, "\n")
