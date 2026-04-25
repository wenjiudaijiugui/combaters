use nalgebra::DMatrix;

use crate::api::CovariateMatrix;
use crate::batch::BatchLevels;
use crate::error::CombatError;

#[derive(Debug, Clone)]
pub(crate) struct DesignFit {
    pub beta: DMatrix<f64>,
    pub fitted: DMatrix<f64>,
    pub covariate_fitted: DMatrix<f64>,
}

pub(crate) fn build_batch_design(
    n_samples: usize,
    levels: &BatchLevels,
    ref_level: Option<usize>,
) -> DMatrix<f64> {
    let mut design = DMatrix::zeros(n_samples, levels.len());
    for sample in 0..n_samples {
        let level = levels.sample_to_level[sample];
        for column in 0..levels.len() {
            if ref_level == Some(column) || level == column {
                design[(sample, column)] = 1.0;
            }
        }
    }
    design
}

pub(crate) fn build_design(
    n_samples: usize,
    levels: &BatchLevels,
    covariates: Option<CovariateMatrix<'_>>,
    ref_level: Option<usize>,
) -> DMatrix<f64> {
    let batch_columns = levels.len();
    let kept_covariates = kept_covariate_columns(n_samples, covariates);
    let mut design = DMatrix::zeros(n_samples, batch_columns + kept_covariates.len());

    let batch_design = build_batch_design(n_samples, levels, ref_level);
    design
        .view_mut((0, 0), (n_samples, batch_columns))
        .copy_from(&batch_design);

    if let Some(covariates) = covariates {
        for (output_column, input_column) in kept_covariates.iter().copied().enumerate() {
            for sample in 0..n_samples {
                design[(sample, batch_columns + output_column)] =
                    covariate_value(covariates, sample, input_column);
            }
        }
    }

    design
}

pub(crate) fn fit_design(
    y: &DMatrix<f64>,
    levels: &BatchLevels,
    covariates: Option<CovariateMatrix<'_>>,
    ref_level: Option<usize>,
) -> Result<DesignFit, CombatError> {
    let design = build_design(y.nrows(), levels, covariates, ref_level);
    let xtx = design.transpose() * &design;
    let Some(xtx_inv) = xtx.try_inverse() else {
        return Err(CombatError::SingularDesign);
    };
    let beta = xtx_inv * design.transpose() * y;
    let fitted = &design * &beta;

    let mut covariate_design = design.clone();
    for sample in 0..covariate_design.nrows() {
        for column in 0..levels.len() {
            covariate_design[(sample, column)] = 0.0;
        }
    }
    let covariate_fitted = covariate_design * &beta;

    Ok(DesignFit {
        beta,
        fitted,
        covariate_fitted,
    })
}

#[cfg(test)]
pub(crate) fn fit_no_covariate_design(
    y: &DMatrix<f64>,
    levels: &BatchLevels,
) -> Result<DesignFit, CombatError> {
    fit_design(y, levels, None, None)
}

fn kept_covariate_columns(n_samples: usize, covariates: Option<CovariateMatrix<'_>>) -> Vec<usize> {
    let Some(covariates) = covariates else {
        return Vec::new();
    };
    (0..covariates.n_covariates)
        .filter(|&column| {
            (0..n_samples).any(|sample| covariate_value(covariates, sample, column) != 1.0)
        })
        .collect()
}

fn covariate_value(covariates: CovariateMatrix<'_>, sample: usize, column: usize) -> f64 {
    covariates.values[sample * covariates.n_covariates + column]
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::batch::BatchLevels;

    use crate::api::CovariateMatrix;

    use super::{build_batch_design, build_design, fit_no_covariate_design};

    #[test]
    fn batch_design_uses_compact_level_columns() {
        let levels = BatchLevels::from_ids(&[10, 20, 10, 30], 4).unwrap();
        let design = build_batch_design(4, &levels, None);

        assert_eq!(design.nrows(), 4);
        assert_eq!(design.ncols(), 3);
        assert_eq!(
            [design[(0, 0)], design[(0, 1)], design[(0, 2)]],
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            [design[(1, 0)], design[(1, 1)], design[(1, 2)]],
            [0.0, 1.0, 0.0]
        );
        assert_eq!(
            [design[(3, 0)], design[(3, 1)], design[(3, 2)]],
            [0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn reference_batch_design_keeps_reference_column_as_intercept() {
        let levels = BatchLevels::from_ids(&[10, 20, 10, 30], 4).unwrap();
        let ref_level = levels.resolve_raw(20);
        let design = build_batch_design(4, &levels, ref_level);

        assert_eq!(design.nrows(), 4);
        assert_eq!(design.ncols(), 3);
        assert_eq!(
            [design[(0, 0)], design[(0, 1)], design[(0, 2)]],
            [1.0, 1.0, 0.0]
        );
        assert_eq!(
            [design[(1, 0)], design[(1, 1)], design[(1, 2)]],
            [0.0, 1.0, 0.0]
        );
        assert_eq!(
            [design[(3, 0)], design[(3, 1)], design[(3, 2)]],
            [0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn no_covariate_beta_is_batch_feature_mean() {
        let levels = BatchLevels::from_ids(&[0, 0, 1, 1], 4).unwrap();
        let y = DMatrix::from_row_slice(4, 2, &[1.0, 10.0, 3.0, 14.0, 5.0, 20.0, 7.0, 24.0]);

        let fit = fit_no_covariate_design(&y, &levels).unwrap();

        assert_eq!(fit.beta[(0, 0)], 2.0);
        assert_eq!(fit.beta[(0, 1)], 12.0);
        assert_eq!(fit.beta[(1, 0)], 6.0);
        assert_eq!(fit.beta[(1, 1)], 22.0);
    }

    #[test]
    fn design_drops_intercept_covariate_column() {
        let levels = BatchLevels::from_ids(&[0, 0, 1, 1], 4).unwrap();
        let covariates = [1.0, 0.1, 1.0, 0.2, 1.0, 0.3, 1.0, 0.4];
        let design = build_design(
            4,
            &levels,
            Some(CovariateMatrix {
                values: &covariates,
                n_covariates: 2,
            }),
            None,
        );

        assert_eq!(design.nrows(), 4);
        assert_eq!(design.ncols(), 3);
        assert_eq!(
            [design[(0, 0)], design[(0, 1)], design[(0, 2)]],
            [1.0, 0.0, 0.1]
        );
        assert_eq!(
            [design[(3, 0)], design[(3, 1)], design[(3, 2)]],
            [0.0, 1.0, 0.4]
        );
    }
}
