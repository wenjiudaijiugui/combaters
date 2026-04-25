use nalgebra::DMatrix;

use crate::api::CovariateMatrix;
use crate::batch::BatchLevels;
use crate::design::fit_design;
use crate::error::CombatError;

#[derive(Debug, Clone)]
pub(crate) struct Standardization {
    pub var_pooled: Vec<f64>,
    pub stand_mean: DMatrix<f64>,
    pub s_data: DMatrix<f64>,
}

pub(crate) fn dense_values_to_matrix(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
) -> DMatrix<f64> {
    DMatrix::from_row_slice(n_samples, n_features, values)
}

pub(crate) fn standardize(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    levels: &BatchLevels,
    covariates: Option<CovariateMatrix<'_>>,
    ref_level: Option<usize>,
) -> Result<Standardization, CombatError> {
    let y = dense_values_to_matrix(values, n_samples, n_features);
    let fit = fit_design(&y, levels, covariates, ref_level)?;

    let mut grand_mean = vec![0.0; n_features];
    if let Some(ref_level) = ref_level {
        for (feature, mean) in grand_mean.iter_mut().enumerate() {
            *mean = fit.beta[(ref_level, feature)];
        }
    } else {
        for (feature, mean) in grand_mean.iter_mut().enumerate() {
            for level in 0..levels.len() {
                let weight = levels.counts[level] as f64 / n_samples as f64;
                *mean += weight * fit.beta[(level, feature)];
            }
        }
    }

    let mut var_pooled = vec![0.0; n_features];
    for sample in 0..n_samples {
        if let Some(ref_level) = ref_level
            && levels.sample_to_level[sample] != ref_level
        {
            continue;
        }
        for feature in 0..n_features {
            let residual = y[(sample, feature)] - fit.fitted[(sample, feature)];
            var_pooled[feature] += residual * residual;
        }
    }

    let variance_denominator = ref_level.map_or(n_samples, |level| levels.counts[level]);
    for variance in &mut var_pooled {
        *variance /= variance_denominator as f64;
        if !variance.is_finite() || *variance <= 0.0 {
            return Err(CombatError::NumericalFailure {
                reason: "pooled variance is not positive finite".to_string(),
            });
        }
    }

    let mut stand_mean = DMatrix::zeros(n_samples, n_features);
    let mut s_data = DMatrix::zeros(n_samples, n_features);
    for sample in 0..n_samples {
        for feature in 0..n_features {
            stand_mean[(sample, feature)] =
                grand_mean[feature] + fit.covariate_fitted[(sample, feature)];
            s_data[(sample, feature)] =
                (y[(sample, feature)] - stand_mean[(sample, feature)]) / var_pooled[feature].sqrt();
        }
    }

    Ok(Standardization {
        var_pooled,
        stand_mean,
        s_data,
    })
}

#[cfg(test)]
pub(crate) fn standardize_no_covariates(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    levels: &BatchLevels,
) -> Result<Standardization, CombatError> {
    standardize(values, n_samples, n_features, levels, None, None)
}

#[cfg(test)]
mod tests {
    use crate::batch::BatchLevels;

    use super::{dense_values_to_matrix, standardize_no_covariates};

    #[test]
    fn dense_values_to_matrix_preserves_row_major_sample_feature_contract() {
        let matrix = dense_values_to_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(0, 1)], 2.0);
        assert_eq!(matrix[(1, 0)], 3.0);
        assert_eq!(matrix[(2, 1)], 6.0);
    }

    #[test]
    fn standardizes_against_weighted_batch_grand_mean() {
        let values = [1.0, 10.0, 3.0, 14.0, 5.0, 20.0, 7.0, 24.0];
        let levels = BatchLevels::from_ids(&[0, 0, 1, 1], 4).unwrap();

        let state = standardize_no_covariates(&values, 4, 2, &levels).unwrap();

        assert!((state.var_pooled[0] - 1.0).abs() <= 1e-12);
        assert!((state.var_pooled[1] - 4.0).abs() <= 1e-12);
        assert_eq!(state.stand_mean[(0, 0)], 4.0);
        assert_eq!(state.stand_mean[(2, 0)], 4.0);
        assert_eq!(state.stand_mean[(3, 1)], 17.0);
        assert!((state.s_data[(0, 0)] + 3.0).abs() <= 1e-12);
        assert!((state.s_data[(3, 1)] - 3.5).abs() <= 1e-12);
    }

    #[test]
    fn standardizes_against_reference_batch_mean_and_variance() {
        let values = [1.0, 10.0, 3.0, 14.0, 5.0, 20.0, 7.0, 24.0];
        let levels = BatchLevels::from_ids(&[10, 10, 20, 20], 4).unwrap();
        let ref_level = levels.resolve_raw(20);

        let state = super::standardize(&values, 4, 2, &levels, None, ref_level).unwrap();

        assert!((state.var_pooled[0] - 1.0).abs() <= 1e-12);
        assert!((state.var_pooled[1] - 4.0).abs() <= 1e-12);
        assert_eq!(state.stand_mean[(0, 0)], 6.0);
        assert_eq!(state.stand_mean[(3, 1)], 22.0);
        assert!((state.s_data[(2, 0)] + 1.0).abs() <= 1e-12);
        assert!((state.s_data[(3, 1)] - 1.0).abs() <= 1e-12);
    }
}
