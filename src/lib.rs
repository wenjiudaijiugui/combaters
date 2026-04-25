use combaters_core::{
    CombatDenseInput, CombatDenseOptions, CombatDenseResult, CombatError, CovariateMatrix,
    combat_dense,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction(name = "combat")]
#[pyo3(signature = (values, batch, r#mod=None, par_prior=true, mean_only=false, ref_batch=None))]
fn combat_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    batch: PyReadonlyArray1<'py, i64>,
    r#mod: Option<PyReadonlyArray2<'py, f64>>,
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
    let mut covariate_values = None;
    let mut n_covariates = 0;
    if let Some(mod_array) = r#mod.as_ref() {
        let mod_view = mod_array.as_array();
        if !mod_view.is_standard_layout() {
            return Err(PyValueError::new_err(
                "mod must be C-contiguous row-major float64 ndarray",
            ));
        }
        let shape = mod_view.shape();
        if shape[0] != n_samples {
            return Err(PyValueError::new_err(format!(
                "mod row count must match values samples: expected {n_samples}, got {}",
                shape[0]
            )));
        }
        n_covariates = shape[1];
        covariate_values = Some(mod_array.as_slice().map_err(|_| {
            PyValueError::new_err("mod must be C-contiguous row-major float64 ndarray")
        })?);
    }
    let covariates = covariate_values.map(|values| CovariateMatrix {
        values,
        n_covariates,
    });

    let result = combat_dense(
        CombatDenseInput {
            values: values_slice,
            n_samples,
            n_features,
            batch: &batch_ids,
            covariates,
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

fn report_dict<'py>(py: Python<'py>, result: &CombatDenseResult) -> PyResult<Bound<'py, PyDict>> {
    let report = PyDict::new(py);
    report.set_item("effective_mean_only", result.report.effective_mean_only)?;
    report.set_item("singleton_batches", &result.report.singleton_batches)?;
    report.set_item(
        "zero_variance_features",
        &result.report.zero_variance_features,
    )?;
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
    m.add_function(wrap_pyfunction!(combat_py, m)?)?;
    Ok(())
}
