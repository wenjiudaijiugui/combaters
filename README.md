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

The current implementation covers dense ComBat with parametric (`par_prior=True`) and non-parametric (`par_prior=False`) empirical Bayes, `mean_only` true or false, optional `ref_batch` by original batch id, and optional numeric `mod`. If any batch has a single sample, ComBat automatically uses effective mean-only adjustment. With `ref_batch`, reference-batch rows are returned unchanged. `prior.plots` and `BPPARAM` are not exposed.

## Parallel Execution

Parallelism is automatic inside the Rust core and is not a Python or R-style `BPPARAM` API. Small matrices stay on the serial path. Larger matrices use Rayon when the matrix has at least 65,536 cells and at least 64 independent feature-by-batch jobs.

The parallel loops write fixed output indices for feature selection, projection, posterior fitting, adjustment, and feature reinsertion, so results are deterministic for the same inputs. For operational testing only, `COMBATERS_PARALLEL=off` forces the serial path and `COMBATERS_PARALLEL=parallel` forces the parallel path; unset or `auto` keeps the size-based policy.

## Rust Layout

- `crates/combaters-core`: pure Rust ComBat core
- `src/lib.rs`: thin PyO3 binding layer
- `combaters/`: Python package wrapper
