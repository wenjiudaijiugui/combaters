# Parameter Difference Report: combaters vs sva::ComBat

## combaters 0.1.0 Supported Parameters

| Parameter | Value | Status |
|-----------|-------|--------|
| par_prior | True or False | parametric fixture parity for True; non-parametric EB supported |
| mean_only | True or False | supported |
| ref_batch | None or original batch id | supported |
| mod | numeric matrix, DataFrame/Series-like data, optional patsy formula, or None | supported; formula requires optional patsy |
| batch | int64 vector | exact parity |

## Unsupported Parameters

| Parameter | sva supports | combaters | Error message |
|-----------|-------------|-----------|---------------|
| prior.plots | Yes | No | not in API |
| BPPARAM (parallel) | Yes | No | not in API |

## Classification

- **supported**: `par_prior=True` and `par_prior=False`, `mean_only` true or false, optional original-id `ref_batch`, optional `mod` as numeric ndarray, DataFrame/Series-like input, or dummy-coded categorical covariates
- **optional dependency path**: `formula` is available through patsy when that package is installed; the Rust core still receives a numeric matrix
- **documented difference**: `prior.plots`, `BPPARAM`
- **unsupported**: plotting/parallel control
