# combaters

`combaters` is a Rust-backed Python package for dense finite ComBat batch-effect correction.

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

`combat` requires `values` to be a C-contiguous row-major `float64` array with shape `(n_samples, n_features)` and `batch` to be a contiguous `int64` vector with length `n_samples`. Negative batch ids are rejected.

Optional `mod` accepts the existing C-contiguous row-major `float64` ndarray path with shape `(n_samples, n_covariates)`. It also accepts pandas `DataFrame`/`Series` inputs and DataFrame-like objects with `to_numpy()`: numeric columns are kept as covariates, and non-numeric columns are dummy-coded with the first observed level dropped as the reference. A `formula` keyword can be used when `patsy` is installed:

```python
result = combat(values, batch, mod=metadata, formula="~ age + C(treatment)")
```

The Rust core still receives a numeric covariate matrix. Formula support is optional and does not make pandas or patsy required runtime dependencies for the ndarray path.

The current implementation covers dense ComBat with parametric (`par_prior=True`) and non-parametric (`par_prior=False`) empirical Bayes, `mean_only` true or false, optional `ref_batch` by original batch id, and optional `mod`. If any batch has a single sample, ComBat automatically uses effective mean-only adjustment. With `ref_batch`, reference-batch rows are returned unchanged. `prior.plots` and `BPPARAM` are not exposed.

## Rust Layout

- `crates/combaters-core`: pure Rust ComBat core
- `src/lib.rs`: thin PyO3 binding layer
- `combaters/`: Python package wrapper
