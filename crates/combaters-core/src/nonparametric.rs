use std::f64::consts::PI;

use nalgebra::DMatrix;

use crate::batch::BatchLevels;
use crate::error::CombatError;
use crate::parametric::{ParametricEstimates, estimate_delta_hat, estimate_gamma_hat};
use crate::standardize::Standardization;

pub(crate) fn fit_nonparametric(
    state: &Standardization,
    levels: &BatchLevels,
    mean_only: bool,
    ref_level: Option<usize>,
) -> Result<ParametricEstimates, CombatError> {
    let n_features = state.s_data.ncols();
    if n_features < 2 {
        return Err(CombatError::NumericalFailure {
            reason: "non-parametric priors require at least two kept features".to_string(),
        });
    }

    let gamma_hat = estimate_gamma_hat(&state.s_data, levels, ref_level)?;
    let delta_hat = if mean_only {
        DMatrix::from_element(levels.len(), n_features, 1.0)
    } else {
        estimate_delta_hat(&state.s_data, levels)?
    };

    let mut gamma_star = DMatrix::zeros(levels.len(), n_features);
    let mut delta_star = DMatrix::zeros(levels.len(), n_features);

    for level in 0..levels.len() {
        let sample_indices = samples_for_level(levels, level);
        for feature in 0..n_features {
            let (gamma, delta) = posterior_feature(
                &state.s_data,
                &sample_indices,
                level,
                feature,
                &gamma_hat,
                &delta_hat,
            )?;
            gamma_star[(level, feature)] = gamma;
            delta_star[(level, feature)] = delta;
        }
    }

    if let Some(ref_level) = ref_level {
        for feature in 0..n_features {
            gamma_star[(ref_level, feature)] = 0.0;
            delta_star[(ref_level, feature)] = 1.0;
        }
    }

    Ok(ParametricEstimates {
        gamma_star,
        delta_star,
    })
}

fn samples_for_level(levels: &BatchLevels, level: usize) -> Vec<usize> {
    levels
        .sample_to_level
        .iter()
        .enumerate()
        .filter_map(|(sample, &sample_level)| (sample_level == level).then_some(sample))
        .collect()
}

fn posterior_feature(
    s_data: &DMatrix<f64>,
    sample_indices: &[usize],
    level: usize,
    feature: usize,
    gamma_hat: &DMatrix<f64>,
    delta_hat: &DMatrix<f64>,
) -> Result<(f64, f64), CombatError> {
    let n_features = s_data.ncols();
    let n = sample_indices.len() as f64;
    let mut log_likelihoods = Vec::with_capacity(n_features - 1);
    let mut prior_features = Vec::with_capacity(n_features - 1);

    for prior_feature in 0..n_features {
        if prior_feature == feature {
            continue;
        }

        let gamma = gamma_hat[(level, prior_feature)];
        let delta = delta_hat[(level, prior_feature)];
        if !gamma.is_finite() || !delta.is_finite() || delta <= 0.0 {
            return Err(CombatError::NumericalFailure {
                reason: "non-parametric prior value is not positive finite".to_string(),
            });
        }

        let mut sum2 = 0.0;
        for &sample in sample_indices {
            let residual = s_data[(sample, feature)] - gamma;
            sum2 += residual * residual;
        }

        let log_likelihood = -0.5 * n * (2.0 * PI * delta).ln() - sum2 / (2.0 * delta);
        if log_likelihood.is_finite() {
            log_likelihoods.push(log_likelihood);
            prior_features.push(prior_feature);
        }
    }

    if log_likelihoods.is_empty() {
        return Err(CombatError::NumericalFailure {
            reason: "non-parametric likelihood has no finite support".to_string(),
        });
    }

    let max_log = log_likelihoods
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut weight_sum = 0.0;
    let mut gamma_sum = 0.0;
    let mut delta_sum = 0.0;
    for (&log_likelihood, &prior_feature) in log_likelihoods.iter().zip(prior_features.iter()) {
        let weight = (log_likelihood - max_log).exp();
        weight_sum += weight;
        gamma_sum += gamma_hat[(level, prior_feature)] * weight;
        delta_sum += delta_hat[(level, prior_feature)] * weight;
    }

    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(CombatError::NumericalFailure {
            reason: "non-parametric likelihood weights are degenerate".to_string(),
        });
    }

    let gamma_star = gamma_sum / weight_sum;
    let delta_star = delta_sum / weight_sum;
    if !gamma_star.is_finite() || !delta_star.is_finite() || delta_star <= 0.0 {
        return Err(CombatError::NumericalFailure {
            reason: "non-parametric posterior produced a non-finite value".to_string(),
        });
    }

    Ok((gamma_star, delta_star))
}

#[cfg(test)]
mod tests {
    use crate::batch::BatchLevels;
    use crate::standardize::standardize_no_covariates;

    use super::fit_nonparametric;

    #[test]
    fn fit_nonparametric_returns_gamma_star_shape_and_positive_delta_star() {
        let values = [
            4.0, 1.0, 7.0, 2.0, 5.0, 1.5, 8.0, 2.5, 6.5, 2.0, 8.8, 3.0, 11.0, 7.0, 2.0, 6.0, 12.5,
            7.5, 2.5, 6.5, 13.0, 8.0, 3.0, 7.0,
        ];
        let levels = BatchLevels::from_ids(&[10, 10, 10, 20, 20, 20], 6).unwrap();
        let state = standardize_no_covariates(&values, 6, 4, &levels).unwrap();
        let estimates = fit_nonparametric(&state, &levels, false, None).unwrap();

        assert_eq!(estimates.gamma_star.nrows(), 2);
        assert_eq!(estimates.gamma_star.ncols(), 4);
        assert!(estimates.delta_star.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn mean_only_returns_unit_delta_star() {
        let values = [
            4.0, 1.0, 7.0, 2.0, 5.0, 1.5, 8.0, 2.5, 6.5, 2.0, 8.8, 3.0, 11.0, 7.0, 2.0, 6.0, 12.5,
            7.5, 2.5, 6.5, 13.0, 8.0, 3.0, 7.0,
        ];
        let levels = BatchLevels::from_ids(&[10, 10, 10, 20, 20, 20], 6).unwrap();
        let state = standardize_no_covariates(&values, 6, 4, &levels).unwrap();
        let estimates = fit_nonparametric(&state, &levels, true, None).unwrap();

        assert!(estimates.delta_star.iter().all(|value| *value == 1.0));
    }
}
