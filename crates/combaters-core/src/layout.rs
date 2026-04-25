use crate::error::CombatError;

pub(crate) fn checked_matrix_len(
    rows: usize,
    cols: usize,
    context: &'static str,
) -> Result<usize, CombatError> {
    rows.checked_mul(cols)
        .ok_or(CombatError::ShapeOverflow { context })
}

pub(crate) fn validate_dense_shape(
    values_len: usize,
    n_samples: usize,
    n_features: usize,
) -> Result<(), CombatError> {
    if n_samples == 0 {
        return Err(CombatError::EmptyInput {
            context: "n_samples",
        });
    }
    if n_features == 0 {
        return Err(CombatError::EmptyInput {
            context: "n_features",
        });
    }
    let expected = checked_matrix_len(n_samples, n_features, "values")?;
    if values_len != expected {
        return Err(CombatError::ShapeMismatch {
            expected,
            actual: values_len,
            context: "values",
        });
    }
    Ok(())
}

pub(crate) fn row_major_index(sample: usize, feature: usize, n_features: usize) -> usize {
    sample * n_features + feature
}

pub(crate) fn validate_dense_finite(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Result<(), CombatError> {
    for sample in 0..n_samples {
        for feature in 0..n_features {
            let idx = row_major_index(sample, feature, n_features);
            if !values[idx].is_finite() {
                return Err(CombatError::NonFiniteValue { sample, feature });
            }
        }
    }
    Ok(())
}
