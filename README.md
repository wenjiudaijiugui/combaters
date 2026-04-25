# combaters

`combaters` is a Rust-backed Python package for dense finite ComBat batch-effect correction.

The public matrix contract is row-major `samples x features`: `values[sample * n_features + feature]`.

## Python API

```python
import numpy as np
from combaters import combat

values = np.asarray(..., dtype=np.float64).reshape((n_samples, n_features))
batch = np.asarray(...)
mod = np.asarray(..., dtype=np.float64).reshape((n_samples, n_covariates))

result = combat(values, batch, mod=mod, par_prior=True, mean_only=False, ref_batch=None)
adjusted = result["adjusted"]
```

`combat` requires `values` to be a C-contiguous row-major `float64` array with shape `(n_samples, n_features)` and optional `mod` to be a C-contiguous row-major `float64` array with shape `(n_samples, n_covariates)`. `batch` is a one-dimensional vector with length `n_samples`; the Python API accepts R factor-like labels such as strings, object/category labels, and negative integers. Non-negative contiguous `int64` batches keep the direct Rust fast path. Other labels are factorized in Python to compact internal ids before entering the Rust core.

The current implementation covers dense ComBat with parametric (`par_prior=True`) and non-parametric (`par_prior=False`) empirical Bayes, `mean_only` true or false, optional `ref_batch` by the same original label used in `batch`, and optional numeric `mod`. If any batch has a single sample, ComBat automatically uses effective mean-only adjustment. With `ref_batch`, reference-batch rows are returned unchanged. `prior.plots` and `BPPARAM` are not exposed.

## Rust Layout

- `crates/combaters-core`: pure Rust ComBat core
- `src/lib.rs`: thin PyO3 binding layer
- `combaters/`: Python package wrapper
