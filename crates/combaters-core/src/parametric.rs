use nalgebra::DMatrix;

use crate::batch::BatchLevels;
use crate::design::{build_batch_design, fit_coefficients_by_feature, fit_complete_coefficients};
use crate::error::CombatError;
use crate::parallel::ParallelPlan;
use crate::standardize::Standardization;

#[derive(Debug, Clone)]
pub(crate) struct ParametricEstimates {
    pub gamma_star: DMatrix<f64>,
    pub delta_star: DMatrix<f64>,
}

#[derive(Debug, Clone, Copy)]
struct ParametricPriors {
    gamma_bar: f64,
    t2: f64,
    a_prior: f64,
    b_prior: f64,
}

struct LevelEstimates {
    gamma_star: Vec<f64>,
    delta_star: Vec<f64>,
}

pub(crate) fn fit_parametric(
    state: &Standardization,
    levels: &BatchLevels,
    mean_only: bool,
    ref_level: Option<usize>,
    parallel: ParallelPlan,
) -> Result<ParametricEstimates, CombatError> {
    let n_features = state.s_data.ncols();
    if n_features < 2 {
        return Err(CombatError::NumericalFailure {
            reason: "parametric priors require at least two kept features".to_string(),
        });
    }

    let gamma_hat = estimate_gamma_hat(&state.s_data, levels, ref_level)?;

    let mut gamma_star = DMatrix::zeros(levels.len(), n_features);
    let mut delta_star = DMatrix::from_element(levels.len(), n_features, 1.0);

    if mean_only {
        let level_estimates = fit_levels(levels.len(), parallel, |level| {
            fit_mean_only_level(&gamma_hat, level)
        });
        write_level_estimates(level_estimates, &mut gamma_star, &mut delta_star)?;
    } else {
        let delta_hat = estimate_delta_hat(&state.s_data, levels)?;
        let level_estimates = fit_levels(levels.len(), parallel, |level| {
            fit_full_level(state, levels, level, &gamma_hat, &delta_hat, parallel)
        });
        write_level_estimates(level_estimates, &mut gamma_star, &mut delta_star)?;
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

pub(crate) fn fit_unshrunken_mean_only(
    state: &Standardization,
    levels: &BatchLevels,
    ref_level: Option<usize>,
) -> Result<ParametricEstimates, CombatError> {
    let mut gamma_star = estimate_gamma_hat(&state.s_data, levels, ref_level)?;
    let delta_star = DMatrix::from_element(levels.len(), state.s_data.ncols(), 1.0);

    if let Some(ref_level) = ref_level {
        for feature in 0..state.s_data.ncols() {
            gamma_star[(ref_level, feature)] = 0.0;
        }
    }

    Ok(ParametricEstimates {
        gamma_star,
        delta_star,
    })
}

fn fit_levels<F>(
    n_levels: usize,
    parallel: ParallelPlan,
    fit_level: F,
) -> Vec<Result<LevelEstimates, CombatError>>
where
    F: Fn(usize) -> Result<LevelEstimates, CombatError> + Send + Sync,
{
    if parallel.should_parallelize(n_levels) {
        use rayon::prelude::*;

        (0..n_levels).into_par_iter().map(fit_level).collect()
    } else {
        (0..n_levels).map(fit_level).collect()
    }
}

fn fit_mean_only_level(
    gamma_hat: &DMatrix<f64>,
    level: usize,
) -> Result<LevelEstimates, CombatError> {
    let n_features = gamma_hat.ncols();
    let gamma_values = row_values(gamma_hat, level);
    let gamma_bar = mean(&gamma_values);
    let t2 = sample_variance(&gamma_values)?;
    if !t2.is_finite() || t2 < 0.0 {
        return Err(CombatError::NumericalFailure {
            reason: "parametric prior variance is not finite".to_string(),
        });
    }

    let gamma_star = (0..n_features)
        .map(|feature| postmean(gamma_hat[(level, feature)], gamma_bar, 1.0, 1.0, t2))
        .collect();
    Ok(LevelEstimates {
        gamma_star,
        delta_star: vec![1.0; n_features],
    })
}

fn fit_full_level(
    state: &Standardization,
    levels: &BatchLevels,
    level: usize,
    gamma_hat: &DMatrix<f64>,
    delta_hat: &DMatrix<f64>,
    parallel: ParallelPlan,
) -> Result<LevelEstimates, CombatError> {
    let gamma_values = row_values(gamma_hat, level);
    let delta_values = row_values(delta_hat, level);
    let gamma_bar = mean(&gamma_values);
    let t2 = sample_variance(&gamma_values)?;
    let delta_mean = mean(&delta_values);
    let delta_var = sample_variance(&delta_values)?;

    if t2 <= 0.0 || delta_var <= 0.0 {
        return Err(CombatError::NumericalFailure {
            reason: "parametric prior variance is not positive".to_string(),
        });
    }

    let a_prior = (2.0 * delta_var + delta_mean * delta_mean) / delta_var;
    let b_prior = (delta_mean * delta_var + delta_mean.powi(3)) / delta_var;
    if !a_prior.is_finite() || !b_prior.is_finite() {
        return Err(CombatError::NumericalFailure {
            reason: "parametric prior is not finite".to_string(),
        });
    }

    let (gamma_star, delta_star) = solve_posterior(
        &state.s_data,
        levels,
        level,
        gamma_hat,
        delta_hat,
        ParametricPriors {
            gamma_bar,
            t2,
            a_prior,
            b_prior,
        },
        parallel,
    )?;

    Ok(LevelEstimates {
        gamma_star,
        delta_star,
    })
}

fn write_level_estimates(
    level_estimates: Vec<Result<LevelEstimates, CombatError>>,
    gamma_star: &mut DMatrix<f64>,
    delta_star: &mut DMatrix<f64>,
) -> Result<(), CombatError> {
    for (level, estimates) in level_estimates.into_iter().enumerate() {
        let estimates = estimates?;
        for feature in 0..gamma_star.ncols() {
            gamma_star[(level, feature)] = estimates.gamma_star[feature];
            delta_star[(level, feature)] = estimates.delta_star[feature];
        }
    }
    Ok(())
}

pub(crate) fn estimate_gamma_hat(
    s_data: &DMatrix<f64>,
    levels: &BatchLevels,
    ref_level: Option<usize>,
) -> Result<DMatrix<f64>, CombatError> {
    let batch_design = build_batch_design(s_data.nrows(), levels, ref_level);
    if s_data.iter().any(|value| value.is_nan()) {
        fit_coefficients_by_feature(s_data, &batch_design)
    } else {
        fit_complete_coefficients(s_data, &batch_design)
    }
}

pub(crate) fn estimate_delta_hat(
    s_data: &DMatrix<f64>,
    levels: &BatchLevels,
) -> Result<DMatrix<f64>, CombatError> {
    let n_features = s_data.ncols();
    let mut delta_hat = DMatrix::<f64>::zeros(levels.len(), n_features);

    for level in 0..levels.len() {
        if levels.counts[level] < 2 {
            return Err(CombatError::UnsupportedOption {
                reason: "scale adjustment requires at least two samples per batch".to_string(),
            });
        }
    }

    for level in 0..levels.len() {
        for feature in 0..n_features {
            let mut count = 0;
            let mut sum = 0.0;

            for sample in 0..s_data.nrows() {
                if levels.sample_to_level[sample] != level {
                    continue;
                }
                let value = s_data[(sample, feature)];
                if value.is_nan() {
                    continue;
                }
                count += 1;
                sum += value;
            }

            if count < 2 {
                return Err(CombatError::UnsupportedOption {
                    reason: "scale adjustment requires at least two observed values per batch and feature"
                        .to_string(),
                });
            }

            let count_f64 = count as f64;
            let mean = sum / count_f64;
            let mut sum_sq = 0.0;
            for sample in 0..s_data.nrows() {
                if levels.sample_to_level[sample] != level {
                    continue;
                }
                let value = s_data[(sample, feature)];
                if value.is_nan() {
                    continue;
                }
                let residual = value - mean;
                sum_sq += residual * residual;
            }

            delta_hat[(level, feature)] = sum_sq / (count_f64 - 1.0);
            if !delta_hat[(level, feature)].is_finite() || delta_hat[(level, feature)] <= 0.0 {
                return Err(CombatError::NumericalFailure {
                    reason: "delta_hat is not positive finite".to_string(),
                });
            }
        }
    }

    Ok(delta_hat)
}

fn solve_posterior(
    s_data: &DMatrix<f64>,
    levels: &BatchLevels,
    level: usize,
    gamma_hat: &DMatrix<f64>,
    delta_hat: &DMatrix<f64>,
    priors: ParametricPriors,
    parallel: ParallelPlan,
) -> Result<(Vec<f64>, Vec<f64>), CombatError> {
    let mut g_old = row_values(gamma_hat, level);
    let mut d_old = row_values(delta_hat, level);

    for _iteration in 0..1000 {
        let updates = posterior_updates(s_data, levels, level, gamma_hat, &d_old, priors, parallel);
        let (g_new, d_new) = split_posterior_updates(updates)?;

        let change = posterior_change(&g_old, &d_old, &g_new, &d_new);
        g_old = g_new;
        d_old = d_new;

        if change <= 1e-4 {
            return Ok((g_old, d_old));
        }
    }

    Err(CombatError::NumericalFailure {
        reason: "parametric posterior solver did not converge in 1000 iterations".to_string(),
    })
}

fn observed_count_for_level_feature(
    s_data: &DMatrix<f64>,
    levels: &BatchLevels,
    level: usize,
    feature: usize,
) -> Result<usize, CombatError> {
    let count = (0..s_data.nrows())
        .filter(|&sample| {
            levels.sample_to_level[sample] == level && !s_data[(sample, feature)].is_nan()
        })
        .count();

    if count == 0 {
        return Err(CombatError::NumericalFailure {
            reason: "posterior update requires at least one observed value".to_string(),
        });
    }

    Ok(count)
}

fn posterior_updates(
    s_data: &DMatrix<f64>,
    levels: &BatchLevels,
    level: usize,
    gamma_hat: &DMatrix<f64>,
    d_old: &[f64],
    priors: ParametricPriors,
    parallel: ParallelPlan,
) -> Vec<Result<(f64, f64), CombatError>> {
    let n_features = s_data.ncols();
    if parallel.should_parallelize(n_features) {
        use rayon::prelude::*;

        (0..n_features)
            .into_par_iter()
            .map(|feature| {
                posterior_update_feature(s_data, levels, level, feature, gamma_hat, d_old, priors)
            })
            .collect()
    } else {
        (0..n_features)
            .map(|feature| {
                posterior_update_feature(s_data, levels, level, feature, gamma_hat, d_old, priors)
            })
            .collect()
    }
}

fn posterior_update_feature(
    s_data: &DMatrix<f64>,
    levels: &BatchLevels,
    level: usize,
    feature: usize,
    gamma_hat: &DMatrix<f64>,
    d_old: &[f64],
    priors: ParametricPriors,
) -> Result<(f64, f64), CombatError> {
    let n = observed_count_for_level_feature(s_data, levels, level, feature)? as f64;
    let gamma = (priors.t2 * n * gamma_hat[(level, feature)] + d_old[feature] * priors.gamma_bar)
        / (priors.t2 * n + d_old[feature]);

    let mut sum2 = 0.0;
    for sample in 0..s_data.nrows() {
        if levels.sample_to_level[sample] == level {
            let value = s_data[(sample, feature)];
            if value.is_nan() {
                continue;
            }
            let residual = value - gamma;
            sum2 += residual * residual;
        }
    }

    let delta = (0.5 * sum2 + priors.b_prior) / (n / 2.0 + priors.a_prior - 1.0);
    if !gamma.is_finite() || !delta.is_finite() || delta <= 0.0 {
        return Err(CombatError::NumericalFailure {
            reason: "posterior update produced a non-finite value".to_string(),
        });
    }

    Ok((gamma, delta))
}

fn split_posterior_updates(
    updates: Vec<Result<(f64, f64), CombatError>>,
) -> Result<(Vec<f64>, Vec<f64>), CombatError> {
    let mut gamma = Vec::with_capacity(updates.len());
    let mut delta = Vec::with_capacity(updates.len());
    for update in updates {
        let (feature_gamma, feature_delta) = update?;
        gamma.push(feature_gamma);
        delta.push(feature_delta);
    }
    Ok((gamma, delta))
}

fn posterior_change(g_old: &[f64], d_old: &[f64], g_new: &[f64], d_new: &[f64]) -> f64 {
    let mut max_change = 0.0;
    for feature in 0..g_old.len() {
        let gamma_change = if g_old[feature] == 0.0 {
            if g_new[feature] == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (g_new[feature] - g_old[feature]).abs() / g_old[feature]
        };
        let delta_change = if d_old[feature] == 0.0 {
            if d_new[feature] == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (d_new[feature] - d_old[feature]).abs() / d_old[feature].abs()
        };
        if gamma_change.is_finite() {
            max_change = f64::max(max_change, gamma_change);
        } else {
            max_change = f64::INFINITY;
        }
        if delta_change.is_finite() {
            max_change = f64::max(max_change, delta_change);
        } else {
            max_change = f64::INFINITY;
        }
    }
    max_change
}

fn row_values(matrix: &DMatrix<f64>, row: usize) -> Vec<f64> {
    (0..matrix.ncols())
        .map(|column| matrix[(row, column)])
        .collect()
}

fn postmean(g_hat: f64, g_bar: f64, n: f64, d_star: f64, t2: f64) -> f64 {
    (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_variance(values: &[f64]) -> Result<f64, CombatError> {
    if values.len() < 2 {
        return Err(CombatError::NumericalFailure {
            reason: "sample variance requires at least two values".to_string(),
        });
    }
    let center = mean(values);
    let sum_sq = values
        .iter()
        .map(|value| {
            let diff = value - center;
            diff * diff
        })
        .sum::<f64>();
    Ok(sum_sq / (values.len() - 1) as f64)
}

#[cfg(test)]
mod tests {
    use crate::batch::BatchLevels;
    use crate::parallel::ParallelPlan;
    use crate::standardize::standardize_no_covariates;

    use super::fit_parametric;

    #[test]
    fn fit_parametric_returns_gamma_star_shape_and_positive_delta_star() {
        let values = [
            4.0, 1.0, 7.0, 2.0, 5.0, 1.5, 8.0, 2.5, 6.5, 2.0, 8.8, 3.0, 11.0, 7.0, 2.0, 6.0, 12.5,
            7.5, 2.5, 6.5, 13.0, 8.0, 3.0, 7.0,
        ];
        let levels = BatchLevels::from_ids(&[10, 10, 10, 20, 20, 20], 6).unwrap();
        let state = standardize_no_covariates(&values, 6, 4, &levels).unwrap();
        let estimates =
            fit_parametric(&state, &levels, false, None, ParallelPlan::serial()).unwrap();

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
            fit_parametric(&state, &levels, true, None, ParallelPlan::serial()).unwrap();

        assert!(estimates.delta_star.iter().all(|value| *value == 1.0));
    }
}
