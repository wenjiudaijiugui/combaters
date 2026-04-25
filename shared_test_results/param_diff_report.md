# Parameter Difference Report: combaters vs sva::ComBat

## combaters 0.1.0 Supported Parameters

| Parameter | Value | Status |
|-----------|-------|--------|
| par_prior | True or False | parametric fixture parity for True; non-parametric EB supported |
| mean_only | True or False | supported |
| ref_batch | None or original batch id | supported |
| mod | numeric matrix or None | supported |
| batch | int64 vector | exact parity |

## Unsupported Parameters

| Parameter | sva supports | combaters | Error message |
|-----------|-------------|-----------|---------------|
| prior.plots | Yes | No | not in API |
| BPPARAM (parallel) | Yes | No | not in API |

## Classification

- **supported**: `par_prior=True` and `par_prior=False`, `mean_only` true or false, optional original-id `ref_batch`, optional numeric `mod`
- **documented difference**: `prior.plots`, `BPPARAM`
- **unsupported**: plotting/parallel control

## Degenerate Input Policy

| Case | combaters behavior |
|------|--------------------|
| zero-variance feature inside any multi-sample batch | copy that feature back unchanged and report its zero-based index in `zero_variance_features` |
| all features are zero-variance/unadjustable | return the original matrix and report all unadjustable feature indexes instead of failing |
| exactly one adjustable feature remains | skip empirical Bayes prior fitting and use unshrunken mean-only location adjustment for that feature |
