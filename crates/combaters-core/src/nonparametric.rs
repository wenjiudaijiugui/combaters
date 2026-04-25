use std::f64::consts::PI;

use nalgebra::DMatrix;

use crate::batch::BatchLevels;
use crate::error::CombatError;
use crate::parallel::ParallelPlan;
use crate::parametric::{ParametricEstimates, estimate_delta_hat, estimate_gamma_hat};
use crate::standardize::Standardization;

pub(crate) fn fit_nonparametric(
    state: &Standardization,
    levels: &BatchLevels,
    mean_only: bool,
    ref_level: Option<usize>,
    parallel: ParallelPlan,
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
    let samples_by_level: Vec<Vec<usize>> = (0..levels.len())
        .map(|level| samples_for_level(levels, level))
        .collect();

    let posteriors = fit_feature_posteriors(
        state,
        levels,
        &samples_by_level,
        &gamma_hat,
        &delta_hat,
        parallel,
    );
    for posterior in posteriors {
        let posterior = posterior?;
        gamma_star[(posterior.level, posterior.feature)] = posterior.gamma;
        delta_star[(posterior.level, posterior.feature)] = posterior.delta;
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

struct FeaturePosterior {
    level: usize,
    feature: usize,
    gamma: f64,
    delta: f64,
}

fn fit_feature_posteriors(
    state: &Standardization,
    levels: &BatchLevels,
    samples_by_level: &[Vec<usize>],
    gamma_hat: &DMatrix<f64>,
    delta_hat: &DMatrix<f64>,
    parallel: ParallelPlan,
) -> Vec<Result<FeaturePosterior, CombatError>> {
    let n_features = state.s_data.ncols();
    let n_jobs = levels.len() * n_features;
    if parallel.should_parallelize(n_jobs) {
        use rayon::prelude::*;

        (0..n_jobs)
            .into_par_iter()
            .map(|job| {
                posterior_job(
                    job,
                    n_features,
                    state,
                    samples_by_level,
                    gamma_hat,
                    delta_hat,
                )
            })
            .collect()
    } else {
        (0..n_jobs)
            .map(|job| {
                posterior_job(
                    job,
                    n_features,
                    state,
                    samples_by_level,
                    gamma_hat,
                    delta_hat,
                )
            })
            .collect()
    }
}

fn posterior_job(
    job: usize,
    n_features: usize,
    state: &Standardization,
    samples_by_level: &[Vec<usize>],
    gamma_hat: &DMatrix<f64>,
    delta_hat: &DMatrix<f64>,
) -> Result<FeaturePosterior, CombatError> {
    let level = job / n_features;
    let feature = job % n_features;
    let (gamma, delta) = posterior_feature(
        &state.s_data,
        &samples_by_level[level],
        level,
        feature,
        gamma_hat,
        delta_hat,
    )?;
    Ok(FeaturePosterior {
        level,
        feature,
        gamma,
        delta,
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
        let mut n = 0;
        for &sample in sample_indices {
            let value = s_data[(sample, feature)];
            if value.is_nan() {
                continue;
            }
            let residual = value - gamma;
            sum2 += residual * residual;
            n += 1;
        }

        let n = n as f64;
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
    use crate::parallel::ParallelPlan;
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
        let estimates =
            fit_nonparametric(&state, &levels, false, None, ParallelPlan::serial()).unwrap();

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
        let estimates =
            fit_nonparametric(&state, &levels, true, None, ParallelPlan::serial()).unwrap();

        assert!(estimates.delta_star.iter().all(|value| *value == 1.0));
    }
}
