# Parameter Difference Report: combaters vs sva::ComBat

## combaters 0.1.0 Supported Parameters

| Parameter | Value | Status |
|-----------|-------|--------|
| par_prior | True or False | parametric fixture parity for True; non-parametric EB supported |
| mean_only | True or False | supported |
| ref_batch | None or original batch label | supported |
| mod | numeric matrix or None | supported |
| batch | R factor-like labels | supports strings, object/category labels, negative integers, and the existing non-negative int64 fast path |
| dat / values missing entries | NA / missing values | supported for `values`; missing coordinates are ignored during fitting and preserved in output |

## Unsupported Parameters

| Parameter | sva supports | combaters | Error message |
|-----------|-------------|-----------|---------------|
| prior.plots | Yes | No | not in API |
| BPPARAM (parallel) | Yes | No public parameter | core auto-selects serial or Rayon parallel execution |

## Classification

- **supported**: `par_prior=True` and `par_prior=False`, `mean_only` true or false, optional original-label `ref_batch`, optional numeric finite `mod`, factor-like `batch` labels, and NA-aware `values`
- **documented difference**: `prior.plots`, `BPPARAM`
- **unsupported**: plotting and user-supplied parallel backends
- **internal behavior**: Rust core keeps small matrices serial and enables Rayon for larger matrices at the documented threshold; `COMBATERS_PARALLEL=off|parallel|auto` is an operational escape hatch, not an API compatibility surface

## Missing Data Notes

`combaters` follows the `sva::ComBat` NA-aware branch for the core data matrix: beta and gamma fits drop missing responses per feature, pooled and batch variances ignore missing values, and parametric/non-parametric posterior updates count only observed standardized values. Infinite data values remain invalid. The optional `mod` matrix is still required to be finite.

## Degenerate Input Policy

| Case | combaters behavior |
|------|--------------------|
| zero-variance feature inside any multi-sample batch | copy that feature back unchanged and report its zero-based index in `zero_variance_features` |
| all features are zero-variance/unadjustable | return the original matrix and report all unadjustable feature indexes instead of failing |
| exactly one adjustable feature remains | skip empirical Bayes prior fitting and use unshrunken mean-only location adjustment for that feature |
