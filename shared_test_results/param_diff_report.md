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
| BPPARAM (parallel) | Yes | No public parameter | core auto-selects serial or Rayon parallel execution |

## Classification

- **supported**: `par_prior=True` and `par_prior=False`, `mean_only` true or false, optional original-id `ref_batch`, optional numeric `mod`
- **documented difference**: `prior.plots`, `BPPARAM`
- **unsupported**: plotting and user-supplied parallel backends
- **internal behavior**: Rust core keeps small matrices serial and enables Rayon for larger matrices at the documented threshold; `COMBATERS_PARALLEL=off|parallel|auto` is an operational escape hatch, not an API compatibility surface
