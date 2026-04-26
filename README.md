# combaters

[![Python](https://img.shields.io/badge/python-3.10--3.14-blue)](https://www.python.org/)
[![PyO3](https://img.shields.io/badge/PyO3-abi3--py310-orange)](https://pyo3.rs/)
[![Rust](https://img.shields.io/badge/core-Rust-dea584)](https://www.rust-lang.org/)
[![sva::ComBat](https://img.shields.io/badge/rewrite-sva%3A%3AComBat-4b8bbe)](https://bioconductor.org/packages/release/bioc/html/sva.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](https://opensource.org/licenses/MIT)

[中文文档](README.zh-CN.md)

`combaters` is a Rust/PyO3 rewrite of Bioconductor `sva::ComBat` for dense
ComBat batch-effect correction in Python. It keeps the familiar ComBat behavior
while moving the numerical core into Rust for predictable packaging, memory use,
and runtime performance.

The public matrix contract is row-major `samples x features`: `values[sample * n_features + feature]`.

## Python Compatibility

Release wheels target CPython 3.10 through 3.14. The extension is built with
PyO3 `abi3-py310`; expand this range only after the pinned PyO3 version supports
building against the newer Python minor version.

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

All public matrix inputs use shape `(n_samples, n_features)`: rows are samples
and columns are features. The Python wrapper converts numeric arrays to
C-contiguous `float64` before entering the Rust core.

### Parameters

| Parameter | Applies to | Description |
| --- | --- | --- |
| `values` | `combat`, `combat_frame` | Samples x features data. Lists, tuples, NumPy arrays, pandas `DataFrame`, and SciPy sparse matrices are accepted by `combat`; sparse matrices are densified. `combat_frame` requires a pandas `DataFrame`. |
| `adata` | `combat_anndata` | AnnData-like object read from `adata.X` or `adata.layers[layer]`; it is not mutated. |
| `batch` | all | One-dimensional labels of length `n_samples`. Strings, negative integers, categories, and strided arrays are accepted. `combat_anndata` also accepts an `obs` column name. |
| `mod` | all | Optional sample covariates with `n_samples` rows. Numeric columns are used directly; non-numeric DataFrame-like columns are dummy-coded with the first observed level dropped. |
| `formula` | all | Optional patsy formula for constructing `mod`; requires optional `patsy` support and `mod` data. |
| `par_prior` | all | `True` uses parametric empirical Bayes; `False` uses non-parametric empirical Bayes. |
| `mean_only` | all | `True` adjusts batch location only. Singleton batches and degenerate feature cases can also force effective mean-only behavior. |
| `ref_batch` | all | Optional reference batch using the original batch label. Reference-batch rows are returned unchanged. |
| `layer` | `combat_anndata` | Optional AnnData layer name. `None` reads `adata.X`. |

### Returns

All entry points return a dictionary with `adjusted`, `n_samples`,
`n_features`, and `report`. `combat` returns a NumPy array for array-like input
and preserves pandas labels when `values` is a `DataFrame`. `combat_frame`
always returns `adjusted` as a `DataFrame`. `combat_anndata` returns a
`DataFrame` when pandas and AnnData labels are available; otherwise it returns a
NumPy array.

```python
from combaters import combat_frame

result = combat_frame(values_df, batch_series)
adjusted_df = result["adjusted"]
```

```python
from combaters import combat_anndata

result = combat_anndata(adata, "batch", layer=None)
adjusted = result["adjusted"]
```

A `formula` keyword can be used when `patsy` is installed:

```python
result = combat(values, batch, mod=metadata, formula="~ age + C(treatment)")
```

Install `combaters[ecosystem]` to pull in the optional pandas and SciPy helpers.

Missing values in `values` are ignored during fitting and preserved in
`adjusted`; infinite values are rejected. Features with zero variance inside any
multi-sample batch are copied unchanged and reported in
`result["report"]["zero_variance_features"]`. `prior.plots` and `BPPARAM` are
not exposed; plotting is not implemented, and parallel execution is automatic
inside the Rust core.

## Parallel Execution

Parallelism is automatic inside the Rust core and is not a Python or R-style `BPPARAM` API. Small matrices stay on the serial path. Larger matrices use Rayon when the matrix has at least 65,536 cells and at least 64 independent feature-by-batch jobs.

The parallel loops write fixed output indices for feature selection, projection, posterior fitting, adjustment, and feature reinsertion, so results are deterministic for the same inputs. For operational testing only, `COMBATERS_PARALLEL=off` forces the serial path and `COMBATERS_PARALLEL=parallel` forces the parallel path; unset or `auto` keeps the size-based policy.

## Rust Layout

- `crates/combaters-core`: pure Rust ComBat core
- `src/lib.rs`: thin PyO3 binding layer
- `combaters/`: Python package wrapper

## Citation

If you use `combaters`, cite the original ComBat method and the Bioconductor
`sva` package that provides the reference `sva::ComBat` implementation:

```bibtex
@article{johnson2007combat,
  title = {Adjusting batch effects in microarray expression data using empirical Bayes methods},
  author = {Johnson, W. Evan and Li, Cheng and Rabinovic, Ariel},
  journal = {Biostatistics},
  volume = {8},
  number = {1},
  pages = {118--127},
  year = {2007},
  doi = {10.1093/biostatistics/kxj037}
}

@article{leek2012sva,
  title = {The sva package for removing batch effects and other unwanted variation in high-throughput experiments},
  author = {Leek, Jeffrey T. and Johnson, W. Evan and Parker, Hilary S. and Jaffe, Andrew E. and Storey, John D.},
  journal = {Bioinformatics},
  volume = {28},
  number = {6},
  pages = {882--883},
  year = {2012},
  doi = {10.1093/bioinformatics/bts034}
}
```

- ComBat method: <https://doi.org/10.1093/biostatistics/kxj037>
- `sva` package: <https://doi.org/10.1093/bioinformatics/bts034>
- Bioconductor `sva`: <https://bioconductor.org/packages/release/bioc/html/sva.html>
