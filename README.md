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

`combat` requires `values` to be a C-contiguous row-major `float64` array with shape `(n_samples, n_features)`, `batch` to be a contiguous `int64` vector with length `n_samples`, and optional `mod` to be a C-contiguous row-major `float64` array with shape `(n_samples, n_covariates)`. Negative batch ids are rejected.

Pandas `DataFrame` input is supported through either `combat_frame(...)` or `combat(...)`.
The returned result keeps the existing dictionary shape, and `result["adjusted"]`
is a `DataFrame` with the original index and columns:

```python
from combaters import combat_frame

result = combat_frame(values_df, batch_series)
adjusted_df = result["adjusted"]
```

Install `combaters[ecosystem]` to pull in the optional pandas and SciPy helpers.

SciPy sparse matrices are accepted by `combat(...)` and are explicitly densified
to a new C-contiguous `float64` NumPy array before the Rust core runs. This makes
the sparse-to-dense copy intentional and predictable, but it can require
substantial memory for large matrices.

AnnData-like objects can use the duck-typed helper without mutating the object:

```python
from combaters import combat_anndata

result = combat_anndata(adata, "batch", layer=None)
adjusted = result["adjusted"]
```

`combat_anndata` reads `adata.X` by default, or `adata.layers[layer]` when a
layer is supplied. A string `batch` is read from `adata.obs[batch]`.

The current implementation covers dense ComBat with parametric (`par_prior=True`) and non-parametric (`par_prior=False`) empirical Bayes, `mean_only` true or false, optional `ref_batch` by original batch id, and optional numeric `mod`. If any batch has a single sample, ComBat automatically uses effective mean-only adjustment. With `ref_batch`, reference-batch rows are returned unchanged. `prior.plots` and `BPPARAM` are not exposed.

## Rust Layout

- `crates/combaters-core`: pure Rust ComBat core
- `src/lib.rs`: thin PyO3 binding layer
- `combaters/`: Python package wrapper
