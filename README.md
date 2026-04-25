# combaters

`combaters` is a Rust-backed Python package for dense ComBat batch-effect correction with optional missing values in the data matrix.

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

`combat` accepts array-like `values` with shape `(n_samples, n_features)`, including lists or tuples, `float32` or integer arrays, and Fortran-order or strided arrays. The Python wrapper converts `values` and optional `mod` to C-contiguous `float64` arrays. `values` may contain `np.nan`/NA entries, which are ignored during fitting and preserved as missing values in the adjusted matrix. Infinite values in `values` are rejected. The optional `mod` matrix must be finite.

`batch` is a one-dimensional vector with length `n_samples`; the Python API accepts R factor-like labels such as strings, object/category labels, negative integers, and strided/integer arrays. Non-negative contiguous `int64` batches keep the direct Rust fast path. Other labels are factorized in Python to compact internal ids before entering the Rust core. `ref_batch` uses the same original label type as `batch`.

Optional `mod` accepts the existing numeric ndarray path with shape `(n_samples, n_covariates)`. It also accepts pandas `DataFrame`/`Series` inputs and DataFrame-like objects with `to_numpy()`: numeric columns are kept as covariates, and non-numeric columns are dummy-coded with the first observed level dropped as the reference. A `formula` keyword can be used when `patsy` is installed:

```python
result = combat(values, batch, mod=metadata, formula="~ age + C(treatment)")
```

The Rust core still receives a numeric covariate matrix. Formula support is optional and does not make pandas or patsy required runtime dependencies for the ndarray path.

Pandas `DataFrame` input is supported through either `combat_frame(...)` or `combat(...)`. The returned result keeps the existing dictionary shape, and `result["adjusted"]` is a `DataFrame` with the original index and columns:

```python
from combaters import combat_frame

result = combat_frame(values_df, batch_series)
adjusted_df = result["adjusted"]
```

Install `combaters[ecosystem]` to pull in the optional pandas and SciPy helpers.

SciPy sparse matrices are accepted by `combat(...)` and are explicitly densified to a new C-contiguous `float64` NumPy array before the Rust core runs. This makes the sparse-to-dense copy intentional and predictable, but it can require substantial memory for large matrices.

AnnData-like objects can use the duck-typed helper without mutating the object:

```python
from combaters import combat_anndata

result = combat_anndata(adata, "batch", layer=None)
adjusted = result["adjusted"]
```

`combat_anndata` reads `adata.X` by default, or `adata.layers[layer]` when a layer is supplied. A string `batch` is read from `adata.obs[batch]`.

The current implementation covers dense ComBat with parametric (`par_prior=True`) and non-parametric (`par_prior=False`) empirical Bayes, `mean_only` true or false, optional `ref_batch` by original batch label, optional `mod`, and NA-aware fitting for missing `values`. If any batch has a single sample, ComBat automatically uses effective mean-only adjustment. With `ref_batch`, reference-batch rows are returned unchanged. Features must still have enough observed values to fit the design and, when scale adjustment is enabled, at least two observed values per batch and feature. `prior.plots` and `BPPARAM` are not exposed.

Degenerate feature handling is user-facing and reported. Features with zero variance inside any multi-sample batch are treated as unadjustable, copied back unchanged, and listed by zero-based column index in `result["report"]["zero_variance_features"]`. If every feature is unadjustable, `adjusted` is the original matrix and no hard failure is raised. If exactly one feature remains adjustable, empirical Bayes prior fitting is skipped and that feature uses unshrunken mean-only location adjustment; `result["report"]["effective_mean_only"]` is `True`.

## Parallel Execution

Parallelism is automatic inside the Rust core and is not a Python or R-style `BPPARAM` API. Small matrices stay on the serial path. Larger matrices use Rayon when the matrix has at least 65,536 cells and at least 64 independent feature-by-batch jobs.

The parallel loops write fixed output indices for feature selection, projection, posterior fitting, adjustment, and feature reinsertion, so results are deterministic for the same inputs. For operational testing only, `COMBATERS_PARALLEL=off` forces the serial path and `COMBATERS_PARALLEL=parallel` forces the parallel path; unset or `auto` keeps the size-based policy.

## Rust Layout

- `crates/combaters-core`: pure Rust ComBat core
- `src/lib.rs`: thin PyO3 binding layer
- `combaters/`: Python package wrapper
