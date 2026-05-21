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

cat("R reference outputs written to", out_dir, "\n")
