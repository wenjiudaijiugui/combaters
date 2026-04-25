# Parameter Difference Report: combaters vs sva::ComBat

## combaters 0.1.0 Supported Parameters

| Parameter | Value | Status |
|-----------|-------|--------|
| par_prior | True or False | parametric fixture parity for True; non-parametric EB supported |
| mean_only | True or False | supported |
| ref_batch | None or original batch label | supported |
| mod | numeric matrix or None | supported |
| batch | R factor-like labels | supports strings, object/category labels, negative integers, and the existing non-negative int64 fast path |

## Unsupported Parameters

| Parameter | sva supports | combaters | Error message |
|-----------|-------------|-----------|---------------|
| prior.plots | Yes | No | not in API |
| BPPARAM (parallel) | Yes | No | not in API |

## Classification

- **supported**: `par_prior=True` and `par_prior=False`, `mean_only` true or false, optional original-label `ref_batch`, optional numeric `mod`, factor-like `batch` labels
- **documented difference**: `prior.plots`, `BPPARAM`
- **unsupported**: plotting/parallel control
