# NumPy Buffer API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a NumPy-backed Python API that lets users pass dense `float64` matrices without converting them to Python lists, while preserving the existing `combat_dense(list)` API.

**Architecture:** Keep `combaters-core` unchanged. Add a PyO3/rust-numpy binding function in the top-level extension crate that reads a C-contiguous `numpy.ndarray` as a borrowed `&[f64]`, converts `int64` batch labels into the existing `usize` batch contract, calls the same Rust core, and returns a 2D `numpy.ndarray` in `result["adjusted"]`. The Python package exports this as `combat_dense_numpy`; the existing `combat_dense` wrapper remains list-based and behavior-compatible.

**Tech Stack:** Rust 2024, PyO3 `0.28.3`, rust-numpy `0.28.0`, maturin, Python `>=3.10`, NumPy, pytest.

---

## Assumptions

- The new API is additive. Do not change `combat_dense(values: list[float], n_samples, n_features, batch: list[int], ...)`.
- The MVP accepts only C-contiguous 2D `np.float64` values and contiguous 1D `np.int64` batch labels.
- The MVP rejects negative batch ids before calling `combaters-core`, because the core currently accepts `usize` batch ids.
- The new API returns the same dict shape as `combat_dense`, except `result["adjusted"]` is a 2D `np.ndarray` with shape `(n_samples, n_features)`.
- `combaters-core` remains the only algorithm implementation. Do not add NumPy-specific algorithm branches.

## File Structure

- Modify `Cargo.toml`: add the Rust `numpy` dependency that matches the existing PyO3 version.
- Modify `pyproject.toml`: declare the Python NumPy runtime dependency.
- Modify `src/lib.rs`: add the `combat_dense_numpy` PyO3 function and small conversion helpers.
- Modify `combaters/__init__.py`: export the new wrapper.
- Modify `combaters/__init__.pyi`: expose the new function to type checkers.
- Create `tests/python/test_numpy_api.py`: verify NumPy output shape/dtype, list parity, and rejected input layout/batch values.
- Modify `README.md`: document both list and NumPy APIs and clarify the supported ComBat parameter slice.

## Task 1: Add Dependency Declarations

**Files:**
- Modify: `Cargo.toml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add Rust dependency**

In `Cargo.toml`, change the `[dependencies]` section to:

```toml
[dependencies]
combaters-core = { version = "0.1.0", path = "crates/combaters-core" }
numpy = "0.28.0"
pyo3 = { version = "0.28.3", features = ["abi3-py310"] }
```

- [ ] **Step 2: Add Python runtime dependency**

In `pyproject.toml`, insert this line after `requires-python = ">=3.10"`:

```toml
dependencies = ["numpy>=1.23"]
```

The top of `[project]` should read:

```toml
[project]
name = "combaters"
version = "0.1.0"
description = "Rust-backed ComBat batch-effect correction for dense biological matrices"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["numpy>=1.23"]
license = "MIT"
```

- [ ] **Step 3: Verify dependency resolution**

Run:

```bash
cargo check --workspace
```

Expected: compilation succeeds far enough to resolve `numpy v0.28.0` together with `pyo3 v0.28.x`. If this fails with a PyO3 version conflict, pin `pyo3 = "0.28.0"` in the top-level `Cargo.toml` and re-run the command.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock pyproject.toml
git commit -m "build: add numpy binding dependencies"
```

If implementing in the current prototype checkout where most files are untracked, ask before running commit commands; otherwise use this as the checkpoint boundary.

## Task 2: Add Failing Python Contract Tests

**Files:**
- Create: `tests/python/test_numpy_api.py`

- [ ] **Step 1: Create NumPy API tests**

Create `tests/python/test_numpy_api.py` with this full content:

```python
from __future__ import annotations

import numpy as np
import pytest


def balanced_values() -> list[float]:
    return [
        4.0,
        1.0,
        7.0,
        2.0,
        5.0,
        1.5,
        8.0,
        2.5,
        6.5,
        2.0,
        8.8,
        3.0,
        11.0,
        7.0,
        2.0,
        6.0,
        12.5,
        7.5,
        2.5,
        6.5,
        13.0,
        8.0,
        3.0,
        7.0,
    ]


def balanced_matrix() -> np.ndarray:
    return np.asarray(balanced_values(), dtype=np.float64).reshape((6, 4))


def balanced_batch() -> np.ndarray:
    return np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64)


def test_combat_dense_numpy_matches_list_api() -> None:
    from combaters import combat_dense, combat_dense_numpy

    matrix = balanced_matrix()
    batch = balanced_batch()

    list_result = combat_dense(
        values=balanced_values(),
        n_samples=6,
        n_features=4,
        batch=[10, 10, 10, 20, 20, 20],
    )
    numpy_result = combat_dense_numpy(matrix, batch)

    adjusted = numpy_result["adjusted"]
    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == (6, 4)
    assert adjusted.dtype == np.float64
    assert numpy_result["n_samples"] == 6
    assert numpy_result["n_features"] == 4
    assert numpy_result["report"] == list_result["report"]
    np.testing.assert_allclose(
        adjusted.ravel(),
        np.asarray(list_result["adjusted"], dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_combat_dense_numpy_rejects_fortran_order_values() -> None:
    from combaters import combat_dense_numpy

    matrix = np.asfortranarray(balanced_matrix())

    with pytest.raises(ValueError, match="C-contiguous row-major"):
        combat_dense_numpy(matrix, balanced_batch())


def test_combat_dense_numpy_rejects_strided_batch() -> None:
    from combaters import combat_dense_numpy

    batch = np.asarray([10, 99, 10, 99, 10, 99, 20, 99, 20, 99, 20, 99], dtype=np.int64)[::2]

    with pytest.raises(ValueError, match="contiguous int64"):
        combat_dense_numpy(balanced_matrix(), batch)


def test_combat_dense_numpy_rejects_negative_batch_id() -> None:
    from combaters import combat_dense_numpy

    batch = balanced_batch()
    batch[1] = -1

    with pytest.raises(ValueError, match="non-negative"):
        combat_dense_numpy(balanced_matrix(), batch)
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
mamba run -n combaters-test env PYTHONNOUSERSITE=1 maturin develop --release
mamba run -n combaters-test env PYTHONNOUSERSITE=1 python -m pytest tests/python/test_numpy_api.py -q
```

Expected: FAIL during import with `cannot import name 'combat_dense_numpy' from 'combaters'`.

- [ ] **Step 3: Commit**

```bash
git add tests/python/test_numpy_api.py
git commit -m "test: add numpy dense API contract"
```

If implementing without commits, record this as the first test checkpoint.

## Task 3: Implement the PyO3 NumPy Binding

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Replace `src/lib.rs` with the NumPy-aware binding**

Replace the full contents of `src/lib.rs` with:

```rust
use combaters_core::{
    CombatDenseInput, CombatDenseOptions, CombatDenseResult, CombatError, combat_dense,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction(name = "combat_dense")]
#[pyo3(signature = (values, n_samples, n_features, batch, par_prior=true, mean_only=false, ref_batch=None))]
#[allow(clippy::too_many_arguments)]
fn combat_dense_py(
    py: Python<'_>,
    values: Vec<f64>,
    n_samples: usize,
    n_features: usize,
    batch: Vec<usize>,
    par_prior: bool,
    mean_only: bool,
    ref_batch: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let result = combat_dense(
        CombatDenseInput {
            values: &values,
            n_samples,
            n_features,
            batch: &batch,
            covariates: None,
        },
        CombatDenseOptions {
            par_prior,
            mean_only,
            ref_batch,
        },
    )
    .map_err(map_combat_error)?;

    let report = report_dict(py, &result)?;
    let output = PyDict::new(py);
    output.set_item("adjusted", result.adjusted)?;
    output.set_item("n_samples", result.n_samples)?;
    output.set_item("n_features", result.n_features)?;
    output.set_item("report", report)?;
    Ok(output.unbind())
}

#[pyfunction(name = "combat_dense_numpy")]
#[pyo3(signature = (values, batch, par_prior=true, mean_only=false, ref_batch=None))]
fn combat_dense_numpy_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    batch: PyReadonlyArray1<'py, i64>,
    par_prior: bool,
    mean_only: bool,
    ref_batch: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let (n_samples, n_features) = {
        let values_view = values.as_array();
        if !values_view.is_standard_layout() {
            return Err(PyValueError::new_err(
                "values must be C-contiguous row-major float64 ndarray",
            ));
        }
        let shape = values_view.shape();
        (shape[0], shape[1])
    };

    let values_slice = values.as_slice().map_err(|_| {
        PyValueError::new_err("values must be C-contiguous row-major float64 ndarray")
    })?;
    let batch_slice = batch
        .as_slice()
        .map_err(|_| PyValueError::new_err("batch must be contiguous int64 ndarray"))?;
    let batch_ids = batch_ids_from_i64(batch_slice)?;

    let result = combat_dense(
        CombatDenseInput {
            values: values_slice,
            n_samples,
            n_features,
            batch: &batch_ids,
            covariates: None,
        },
        CombatDenseOptions {
            par_prior,
            mean_only,
            ref_batch,
        },
    )
    .map_err(map_combat_error)?;

    let n_samples_out = result.n_samples;
    let n_features_out = result.n_features;
    let report = report_dict(py, &result)?;
    let adjusted = Array2::from_shape_vec((n_samples_out, n_features_out), result.adjusted)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to shape adjusted matrix: {err}")))?
        .into_pyarray(py);

    let output = PyDict::new(py);
    output.set_item("adjusted", adjusted)?;
    output.set_item("n_samples", n_samples_out)?;
    output.set_item("n_features", n_features_out)?;
    output.set_item("report", report)?;
    Ok(output.unbind())
}

fn batch_ids_from_i64(batch: &[i64]) -> PyResult<Vec<usize>> {
    let mut ids = Vec::with_capacity(batch.len());
    for (sample, raw) in batch.iter().copied().enumerate() {
        let id = usize::try_from(raw).map_err(|_| {
            PyValueError::new_err(format!(
                "batch ids must be non-negative; sample {sample} has {raw}"
            ))
        })?;
        ids.push(id);
    }
    Ok(ids)
}

fn report_dict<'py>(
    py: Python<'py>,
    result: &CombatDenseResult,
) -> PyResult<Bound<'py, PyDict>> {
    let report = PyDict::new(py);
    report.set_item("effective_mean_only", result.report.effective_mean_only)?;
    report.set_item("singleton_batches", &result.report.singleton_batches)?;
    report.set_item("zero_variance_features", &result.report.zero_variance_features)?;
    Ok(report)
}

fn map_combat_error(error: CombatError) -> PyErr {
    match error {
        CombatError::SingularDesign | CombatError::NumericalFailure { .. } => {
            PyRuntimeError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

#[pymodule]
fn _combaters(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(combat_dense_py, m)?)?;
    m.add_function(wrap_pyfunction!(combat_dense_numpy_py, m)?)?;
    Ok(())
}
```

- [ ] **Step 2: Run Rust compile check**

Run:

```bash
cargo check --workspace
```

Expected: PASS. If `values.as_array()` or `values.as_slice()` needs trait imports in the local rust-numpy build, add the exact compiler-suggested imports from `numpy` and re-run.

- [ ] **Step 3: Run Rust tests**

Run:

```bash
cargo test --workspace
```

Expected: PASS with the existing Rust unit and integration tests.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs Cargo.lock
git commit -m "feat: add numpy-backed dense binding"
```

If implementing without commits, record this as the Rust binding checkpoint.

## Task 4: Export the Python Wrapper and Stub

**Files:**
- Modify: `combaters/__init__.py`
- Modify: `combaters/__init__.pyi`

- [ ] **Step 1: Replace `combaters/__init__.py`**

Replace the full contents of `combaters/__init__.py` with:

```python
from __future__ import annotations

from ._combaters import combat_dense as _combat_dense
from ._combaters import combat_dense_numpy as _combat_dense_numpy


def combat_dense(
    values: list[float],
    n_samples: int,
    n_features: int,
    batch: list[int],
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]:
    """Run dense parametric ComBat on a row-major samples x features list."""
    return _combat_dense(
        values,
        n_samples,
        n_features,
        batch,
        par_prior,
        mean_only,
        ref_batch,
    )


def combat_dense_numpy(
    values: object,
    batch: object,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]:
    """Run dense parametric ComBat on a C-contiguous float64 NumPy matrix."""
    return _combat_dense_numpy(values, batch, par_prior, mean_only, ref_batch)


__all__ = ["combat_dense", "combat_dense_numpy"]
```

- [ ] **Step 2: Replace `combaters/__init__.pyi`**

Replace the full contents of `combaters/__init__.pyi` with:

```python
from __future__ import annotations

from typing import Any


def combat_dense(
    values: list[float],
    n_samples: int,
    n_features: int,
    batch: list[int],
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]: ...


def combat_dense_numpy(
    values: Any,
    batch: Any,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]: ...
```

- [ ] **Step 3: Run Python tests**

Run:

```bash
mamba run -n combaters-test env PYTHONNOUSERSITE=1 maturin develop --release
mamba run -n combaters-test env PYTHONNOUSERSITE=1 python -m pytest tests/python -q
```

Expected: PASS for `tests/python/test_binding_contract.py` and `tests/python/test_numpy_api.py`.

- [ ] **Step 4: Commit**

```bash
git add combaters/__init__.py combaters/__init__.pyi
git commit -m "feat: export numpy dense API"
```

If implementing without commits, record this as the Python export checkpoint.

## Task 5: Document the New API Contract

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README API section**

In `README.md`, replace the current `## Python API` section with:

```markdown
## Python API

Small inputs can use the list-based API:

```python
from combaters import combat_dense

result = combat_dense(
    values=[...],
    n_samples=6,
    n_features=4,
    batch=[10, 10, 10, 20, 20, 20],
)

adjusted = result["adjusted"]
```

Real dense matrices should use the NumPy API to avoid converting every value into a Python float object:

```python
import numpy as np
from combaters import combat_dense_numpy

values = np.asarray(..., dtype=np.float64).reshape((n_samples, n_features))
batch = np.asarray(..., dtype=np.int64)

result = combat_dense_numpy(values, batch)
adjusted = result["adjusted"]
```

`combat_dense_numpy` requires `values` to be a C-contiguous row-major `float64` array with shape `(n_samples, n_features)` and `batch` to be a contiguous `int64` vector with length `n_samples`. Negative batch ids are rejected.

The current implementation covers the validated dense parametric slice: `par_prior=True`, `mean_only=False`, `ref_batch=None`, no covariates, no singleton batches, and no non-parametric empirical Bayes.
```

Keep the existing `## Rust Layout` section after this replacement.

- [ ] **Step 2: Run docs-adjacent import smoke test**

Run:

```bash
mamba run -n combaters-test env PYTHONNOUSERSITE=1 python - <<'PY'
import numpy as np
from combaters import combat_dense_numpy

values = np.array(
    [
        [4.0, 1.0, 7.0, 2.0],
        [5.0, 1.5, 8.0, 2.5],
        [6.5, 2.0, 8.8, 3.0],
        [11.0, 7.0, 2.0, 6.0],
        [12.5, 7.5, 2.5, 6.5],
        [13.0, 8.0, 3.0, 7.0],
    ],
    dtype=np.float64,
)
batch = np.array([10, 10, 10, 20, 20, 20], dtype=np.int64)
result = combat_dense_numpy(values, batch)
assert result["adjusted"].shape == values.shape
assert result["adjusted"].dtype == np.float64
print("ok")
PY
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document numpy dense API"
```

If implementing without commits, record this as the documentation checkpoint.

## Task 6: Final Verification

**Files:**
- No source edits.

- [ ] **Step 1: Run full Rust verification**

```bash
cargo test --workspace
```

Expected: PASS.

- [ ] **Step 2: Run full Python verification in the project test environment**

```bash
mamba run -n combaters-test env PYTHONNOUSERSITE=1 maturin develop --release
mamba run -n combaters-test env PYTHONNOUSERSITE=1 python -m pytest tests/python -q
```

Expected: PASS.

- [ ] **Step 3: Confirm no returned external-test artifacts were modified**

```bash
git status --short shared_test_results
```

Expected: no modified or deleted files under `shared_test_results/`.

- [ ] **Step 4: Inspect final diff**

```bash
git diff -- Cargo.toml pyproject.toml src/lib.rs combaters/__init__.py combaters/__init__.pyi tests/python/test_numpy_api.py README.md
```

Expected: diff contains only the dependency declarations, NumPy binding, Python export/stub, tests, and README API update described in this plan.

## Self-Review

- Spec coverage: The plan adds a NumPy/buffer API without changing the existing list API, keeps the core algorithm unchanged, rejects unsupported layouts and negative batch ids, and documents the public contract.
- Placeholder scan: No deferred implementation markers are present; every code-changing step includes concrete code.
- Type consistency: The Rust binding accepts `PyReadonlyArray2<f64>` and `PyReadonlyArray1<i64>`; the Python wrapper accepts generic Python objects and documents NumPy requirements; tests pass `np.float64` and `np.int64` arrays.
- Scope check: This is one subsystem, the Python/Rust data boundary. It does not add covariates, reference batches, mean-only, non-parametric EB, or new core algorithm behavior.
