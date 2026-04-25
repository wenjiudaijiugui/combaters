use crate::batch::BatchLevels;
use crate::layout::row_major_index;
use crate::parallel::ParallelPlan;
use crate::parametric::ParametricEstimates;
use crate::standardize::Standardization;

pub(crate) fn adjust_parametric(
    state: &Standardization,
    estimates: &ParametricEstimates,
    levels: &BatchLevels,
    parallel: ParallelPlan,
) -> Vec<f64> {
    let n_samples = state.s_data.nrows();
    let n_features = state.s_data.ncols();
    let var_sqrt: Vec<f64> = state.var_pooled.iter().map(|value| value.sqrt()).collect();
    let mut output = vec![0.0; n_samples * n_features];

    if parallel.should_parallelize(n_samples) {
        use rayon::prelude::*;

        output
            .par_chunks_mut(n_features)
            .enumerate()
            .for_each(|(sample, row)| {
                adjust_sample_row(sample, row, state, estimates, levels, &var_sqrt);
            });
    } else {
        for sample in 0..n_samples {
            let start = row_major_index(sample, 0, n_features);
            adjust_sample_row(
                sample,
                &mut output[start..start + n_features],
                state,
                estimates,
                levels,
                &var_sqrt,
            );
        }
    }

    output
}

fn adjust_sample_row(
    sample: usize,
    output: &mut [f64],
    state: &Standardization,
    estimates: &ParametricEstimates,
    levels: &BatchLevels,
    var_sqrt: &[f64],
) {
    let level = levels.sample_to_level[sample];
    for feature in 0..output.len() {
        let adjusted_standardized = (state.s_data[(sample, feature)]
            - estimates.gamma_star[(level, feature)])
            / estimates.delta_star[(level, feature)].sqrt();
        output[feature] =
            adjusted_standardized * var_sqrt[feature] + state.stand_mean[(sample, feature)];
    }
}
