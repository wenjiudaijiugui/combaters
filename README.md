# combaters

`combaters` is a Rust-backed Python package for dense ComBat batch-effect correction with optional missing values in the data matrix.

The public matrix contract is row-major `samples x features`: `values[sample * n_features + feature]`.

## Python API

```python
import numpy as np
from combaters import combat

values = np.asarray(..., dtype=np.float64).reshape((n_samples, n_features))
batch = np.asarray(..., dtype=np.int64)
mod = np.asarray(..., dtype=np.float64).reshape((n_samples, n_covariates))

result = combat(values, batch, mod=mod, par_prior=True, mean_only=False, ref_batch=None)
adjusted = result["adjusted"]
```

`combat` requires `values` to be a C-contiguous row-major `float64` array with shape `(n_samples, n_features)`, `batch` to be a contiguous `int64` vector with length `n_samples`, and optional `mod` to be a C-contiguous row-major finite `float64` array with shape `(n_samples, n_covariates)`. `values` may contain `np.nan`/NA entries, which are ignored during fitting and preserved as missing values in the adjusted matrix. Infinite values in `values` are rejected. Negative batch ids are rejected.

The current implementation covers dense ComBat with parametric (`par_prior=True`) and non-parametric (`par_prior=False`) empirical Bayes, `mean_only` true or false, optional `ref_batch` by original batch id, optional numeric `mod`, and NA-aware fitting for missing `values`. If any batch has a single sample, ComBat automatically uses effective mean-only adjustment. With `ref_batch`, reference-batch rows are returned unchanged. Features must still have enough observed values to fit the design and, when scale adjustment is enabled, at least two observed values per batch and feature. `prior.plots` and `BPPARAM` are not exposed.

Degenerate feature handling is user-facing and reported. Features with zero variance inside any multi-sample batch are treated as unadjustable, copied back unchanged, and listed by zero-based column index in `result["report"]["zero_variance_features"]`. If every feature is unadjustable, `adjusted` is the original matrix and no hard failure is raised. If exactly one feature remains adjustable, empirical Bayes prior fitting is skipped and that feature uses unshrunken mean-only location adjustment; `result["report"]["effective_mean_only"]` is `True`.

## Rust Layout

- `crates/combaters-core`: pure Rust ComBat core
- `src/lib.rs`: thin PyO3 binding layer
- `combaters/`: Python package wrapper
