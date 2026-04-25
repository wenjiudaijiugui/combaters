args <- commandArgs(trailingOnly = TRUE)
fixture <- NULL
for (i in seq_along(args)) {
  if (identical(args[[i]], "--fixture") && i < length(args)) {
    fixture <- args[[i + 1]]
  }
}

if (!identical(fixture, "balanced_two_batch_parametric")) {
  stop("expected --fixture balanced_two_batch_parametric", call. = FALSE)
}

suppressPackageStartupMessages({
  library(sva)
  library(BiocParallel)
})

root <- file.path("oracle", "fixtures", fixture)
dir.create(root, recursive = TRUE, showWarnings = FALSE)

dat <- matrix(
  c(
    4.0, 5.0, 6.5, 11.0, 12.5, 13.0,
    1.0, 1.5, 2.0, 7.0, 7.5, 8.0,
    7.0, 8.0, 8.8, 2.0, 2.5, 3.0,
    2.0, 2.5, 3.0, 6.0, 6.5, 7.0
  ),
  nrow = 4,
  byrow = TRUE
)
batch <- c(10, 10, 10, 20, 20, 20)

adjusted <- sva::ComBat(
  dat = dat,
  batch = batch,
  mod = NULL,
  par.prior = TRUE,
  prior.plots = FALSE,
  mean.only = FALSE,
  ref.batch = NULL,
  BPPARAM = BiocParallel::SerialParam()
)

write.table(
  t(dat),
  file = file.path(root, "input_samples_x_features.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = FALSE
)
write.table(
  t(adjusted),
  file = file.path(root, "expected_samples_x_features.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = FALSE
)
write.table(
  batch,
  file = file.path(root, "batch.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = FALSE
)

writeLines(capture.output(sessionInfo()), file.path(root, "sessionInfo.txt"))

toml_string <- function(value) {
  escaped <- gsub("\\\\", "\\\\\\\\", value)
  escaped <- gsub("\"", "\\\\\"", escaped)
  paste0("\"", escaped, "\"")
}

bioc_version <- if (requireNamespace("BiocManager", quietly = TRUE)) {
  as.character(BiocManager::version())
} else {
  NA_character_
}

manifest <- c(
  paste0("fixture = ", toml_string(fixture)),
  "orientation = \"samples x features\"",
  paste0("n_samples = ", ncol(dat)),
  paste0("n_features = ", nrow(dat)),
  paste0("batch = [", paste(batch, collapse = ", "), "]"),
  "par_prior = true",
  "mean_only = false",
  "ref_batch = \"none\"",
  "covariates = \"none\"",
  "missing = \"none\"",
  "parity = \"exact\"",
  "abs_tol = 1e-8",
  "rel_tol = 1e-10",
  "allowed_mismatch_count = 0",
  paste0("r_version = ", toml_string(R.version.string)),
  paste0("sva_version = ", toml_string(as.character(utils::packageVersion("sva")))),
  paste0("bioconductor_version = ", toml_string(bioc_version)),
  paste0(
    "fixture_generation_command = ",
    toml_string("mamba run -n combat-ref Rscript oracle/generate_fixtures.R --fixture balanced_two_batch_parametric")
  )
)
writeLines(manifest, file.path(root, "manifest.toml"))
