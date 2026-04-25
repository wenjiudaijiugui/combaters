use crate::batch::BatchLevels;
use crate::error::CombatError;
use crate::layout::{
    checked_matrix_len, row_major_index, validate_dense_finite, validate_dense_shape,
};

#[derive(Debug, Clone, Copy)]
pub struct CombatDenseInput<'a> {
    pub values: &'a [f64],
    pub n_samples: usize,
    pub n_features: usize,
    pub batch: &'a [usize],
    pub covariates: Option<CovariateMatrix<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub struct CovariateMatrix<'a> {
    pub values: &'a [f64],
    pub n_covariates: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct CombatDenseOptions {
    pub par_prior: bool,
    pub mean_only: bool,
    pub ref_batch: Option<usize>,
}

impl Default for CombatDenseOptions {
    fn default() -> Self {
        Self {
            par_prior: true,
            mean_only: false,
            ref_batch: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CombatDenseResult {
    pub adjusted: Vec<f64>,
    pub n_samples: usize,
    pub n_features: usize,
    pub report: CombatDenseReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombatDenseReport {
    pub effective_mean_only: bool,
    pub singleton_batches: Vec<usize>,
    pub zero_variance_features: Vec<usize>,
}

pub fn combat_dense(
    input: CombatDenseInput<'_>,
    options: CombatDenseOptions,
) -> Result<CombatDenseResult, CombatError> {
    let levels = validate_dense_input(&input, options)?;
    crate::dense::combat_dense_impl(input, options, levels)
}

fn validate_dense_input(
    input: &CombatDenseInput<'_>,
    options: CombatDenseOptions,
) -> Result<BatchLevels, CombatError> {
    validate_dense_shape(input.values.len(), input.n_samples, input.n_features)?;
    validate_dense_finite(input.values, input.n_samples, input.n_features)?;

    let levels = BatchLevels::from_ids(input.batch, input.n_samples)?;

    if let Some(raw_ref) = options.ref_batch
        && levels.resolve_raw(raw_ref).is_none()
    {
        return Err(CombatError::MissingReferenceBatch { requested: raw_ref });
    }

    if let Some(covariates) = input.covariates {
        validate_covariates(input.n_samples, &levels, covariates)?;
    }

    Ok(levels)
}

fn validate_covariates(
    n_samples: usize,
    levels: &BatchLevels,
    covariates: CovariateMatrix<'_>,
) -> Result<(), CombatError> {
    let expected = checked_matrix_len(n_samples, covariates.n_covariates, "covariates")?;
    if covariates.values.len() != expected {
        return Err(CombatError::CovariateShapeMismatch {
            n_samples,
            n_covariates: covariates.n_covariates,
            len: covariates.values.len(),
        });
    }

    for sample in 0..n_samples {
        for column in 0..covariates.n_covariates {
            let idx = row_major_index(sample, column, covariates.n_covariates);
            if !covariates.values[idx].is_finite() {
                return Err(CombatError::NonFiniteDesignValue {
                    sample,
                    column,
                    context: "covariates",
                });
            }
        }
    }

    let kept_columns: Vec<usize> = (0..covariates.n_covariates)
        .filter(|&column| !covariate_column_is_intercept(n_samples, covariates, column))
        .collect();

    if levels.len() + kept_columns.len() > n_samples {
        return Err(CombatError::InvalidDesign {
            reason: format!(
                "too many design columns: batches={} covariates={} samples={}",
                levels.len(),
                kept_columns.len(),
                n_samples
            ),
        });
    }

    for (left_pos, &left) in kept_columns.iter().enumerate() {
        for &right in kept_columns.iter().skip(left_pos + 1) {
            let duplicate = (0..n_samples).all(|sample| {
                covariate_value(covariates, sample, left)
                    == covariate_value(covariates, sample, right)
            });
            if duplicate {
                return Err(CombatError::InvalidDesign {
                    reason: format!("duplicate covariate columns {left} and {right}"),
                });
            }
        }
    }

    for column in kept_columns {
        if column_is_constant_within_batches(n_samples, levels, covariates, column) {
            return Err(CombatError::InvalidDesign {
                reason: format!("covariate column {column} is collinear with batch indicators"),
            });
        }
    }

    Ok(())
}

fn covariate_value(covariates: CovariateMatrix<'_>, sample: usize, column: usize) -> f64 {
    let idx = row_major_index(sample, column, covariates.n_covariates);
    covariates.values[idx]
}

fn covariate_column_is_intercept(
    n_samples: usize,
    covariates: CovariateMatrix<'_>,
    column: usize,
) -> bool {
    (0..n_samples).all(|sample| covariate_value(covariates, sample, column) == 1.0)
}

fn column_is_constant_within_batches(
    n_samples: usize,
    levels: &BatchLevels,
    covariates: CovariateMatrix<'_>,
    column: usize,
) -> bool {
    let mut first_values = vec![None; levels.len()];
    for sample in 0..n_samples {
        let level = levels.sample_to_level[sample];
        let value = covariate_value(covariates, sample, column);
        match first_values[level] {
            Some(first) if first != value => return false,
            Some(_) => {}
            None => first_values[level] = Some(value),
        }
    }
    true
}
