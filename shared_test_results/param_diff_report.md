# Parameter Difference Report: combaters vs sva::ComBat

## combaters 0.1.0 Supported Parameters

| Parameter | Value | Status |
|-----------|-------|--------|
| par_prior | True or False | parametric fixture parity for True; non-parametric EB supported |
| mean_only | True or False | supported |
| ref_batch | None or original batch id | supported |
| mod | numeric matrix or None | supported |
| batch | int64 vector | exact parity |
| dat / values missing entries | NA / missing values | supported for `values`; missing coordinates are ignored during fitting and preserved in output |

## Unsupported Parameters

| Parameter | sva supports | combaters | Error message |
|-----------|-------------|-----------|---------------|
| prior.plots | Yes | No | not in API |
| BPPARAM (parallel) | Yes | No | not in API |

## Classification

- **supported**: `par_prior=True` and `par_prior=False`, `mean_only` true or false, optional original-id `ref_batch`, optional numeric finite `mod`, and NA-aware `values`
- **documented difference**: `prior.plots`, `BPPARAM`
- **unsupported**: plotting/parallel control

## Missing Data Notes

`combaters` follows the `sva::ComBat` NA-aware branch for the core data matrix: beta and gamma fits drop missing responses per feature, pooled and batch variances ignore missing values, and parametric/non-parametric posterior updates count only observed standardized values. Infinite data values remain invalid. The optional `mod` matrix is still required to be finite.
