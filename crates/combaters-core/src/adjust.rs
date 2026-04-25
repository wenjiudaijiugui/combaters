use nalgebra::DMatrix;

use crate::batch::BatchLevels;
use crate::parametric::ParametricEstimates;
use crate::standardize::Standardization;

pub(crate) fn adjust_parametric(
    state: &Standardization,
    estimates: &ParametricEstimates,
    levels: &BatchLevels,
) -> Vec<f64> {
    let n_samples = state.s_data.nrows();
    let n_features = state.s_data.ncols();
    let mut adjusted = DMatrix::zeros(n_samples, n_features);

    for sample in 0..n_samples {
        let level = levels.sample_to_level[sample];
        for feature in 0..n_features {
            let adjusted_standardized = (state.s_data[(sample, feature)]
                - estimates.gamma_star[(level, feature)])
                / estimates.delta_star[(level, feature)].sqrt();
            adjusted[(sample, feature)] = adjusted_standardized * state.var_pooled[feature].sqrt()
                + state.stand_mean[(sample, feature)];
        }
    }

    let mut output = Vec::with_capacity(n_samples * n_features);
    for sample in 0..n_samples {
        for feature in 0..n_features {
            output.push(adjusted[(sample, feature)]);
        }
    }
    output
}
