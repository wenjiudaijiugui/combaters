use crate::adjust::adjust_parametric;
use crate::api::{CombatDenseInput, CombatDenseOptions, CombatDenseReport, CombatDenseResult};
use crate::batch::BatchLevels;
use crate::error::CombatError;
use crate::layout::row_major_index;
use crate::nonparametric::fit_nonparametric;
use crate::parallel::ParallelPlan;
use crate::parametric::fit_parametric;
use crate::standardize::standardize;
use crate::zero_variance::{project_features, reinsert_features, select_nonzero_variance_features};

pub(crate) fn combat_dense_impl(
    input: CombatDenseInput<'_>,
    options: CombatDenseOptions,
    levels: BatchLevels,
) -> Result<CombatDenseResult, CombatError> {
    let parallel = ParallelPlan::for_shape(input.n_samples, input.n_features, levels.len());
    combat_dense_impl_with_parallel(input, options, levels, parallel)
}

fn combat_dense_impl_with_parallel(
    input: CombatDenseInput<'_>,
    options: CombatDenseOptions,
    levels: BatchLevels,
    parallel: ParallelPlan,
) -> Result<CombatDenseResult, CombatError> {
    let singleton_batches = levels.singleton_raw_ids();
    let effective_mean_only = options.mean_only || !singleton_batches.is_empty();
    let ref_level = options.ref_batch.and_then(|raw| levels.resolve_raw(raw));

    let selection = select_nonzero_variance_features(
        input.values,
        input.n_samples,
        input.n_features,
        &levels,
        parallel,
    );
    if selection.kept_features.is_empty() {
        return Err(CombatError::NumericalFailure {
            reason: "all features have zero variance within at least one multi-sample batch"
                .to_string(),
        });
    }

    let fitting_values = project_features(
        input.values,
        input.n_samples,
        input.n_features,
        &selection.kept_features,
        parallel,
    );
    let fitting_features = selection.kept_features.len();
    let state = standardize(
        &fitting_values,
        input.n_samples,
        fitting_features,
        &levels,
        input.covariates,
        ref_level,
    )?;
    let estimates = if options.par_prior {
        fit_parametric(&state, &levels, effective_mean_only, ref_level, parallel)?
    } else {
        fit_nonparametric(&state, &levels, effective_mean_only, ref_level, parallel)?
    };
    let adjusted_kept = adjust_parametric(&state, &estimates, &levels, parallel);
    let mut adjusted = reinsert_features(
        input.values,
        &adjusted_kept,
        input.n_samples,
        input.n_features,
        &selection.kept_features,
        parallel,
    );
    if let Some(ref_level) = ref_level {
        preserve_reference_batch(
            input.values,
            &mut adjusted,
            input.n_samples,
            input.n_features,
            &levels,
            ref_level,
            parallel,
        );
    }

    Ok(CombatDenseResult {
        adjusted,
        n_samples: input.n_samples,
        n_features: input.n_features,
        report: CombatDenseReport {
            effective_mean_only,
            singleton_batches,
            zero_variance_features: selection.zero_variance_features,
        },
    })
}

fn preserve_reference_batch(
    original: &[f64],
    adjusted: &mut [f64],
    n_samples: usize,
    n_features: usize,
    levels: &BatchLevels,
    ref_level: usize,
    parallel: ParallelPlan,
) {
    if parallel.should_parallelize(n_samples) {
        use rayon::prelude::*;

        adjusted
            .par_chunks_mut(n_features)
            .enumerate()
            .for_each(|(sample, row)| {
                if levels.sample_to_level[sample] == ref_level {
                    let start = row_major_index(sample, 0, n_features);
                    row.copy_from_slice(&original[start..start + n_features]);
                }
            });
    } else {
        for sample in 0..n_samples {
            if levels.sample_to_level[sample] != ref_level {
                continue;
            }
            for feature in 0..n_features {
                let idx = row_major_index(sample, feature, n_features);
                adjusted[idx] = original[idx];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::api::{CombatDenseInput, CombatDenseOptions};
    use crate::batch::BatchLevels;
    use crate::parallel::ParallelPlan;

    use super::combat_dense_impl_with_parallel;

    fn deterministic_matrix(n_samples: usize, n_features: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(n_samples * n_features);
        for sample in 0..n_samples {
            let batch_shift = if sample < n_samples / 2 { 0.0 } else { 3.0 };
            let within_batch = (sample % (n_samples / 2)) as f64;
            for feature in 0..n_features {
                let signal = feature as f64 * 0.03;
                let jitter = ((sample * 31 + feature * 17) % 11) as f64 * 0.007;
                values.push(batch_shift + within_batch * 0.19 + signal + jitter);
            }
        }
        values
    }

    fn two_batch_labels(n_samples: usize) -> Vec<usize> {
        (0..n_samples)
            .map(|sample| if sample < n_samples / 2 { 10 } else { 20 })
            .collect()
    }

    fn run_with_plan(
        values: &[f64],
        batch: &[usize],
        n_samples: usize,
        n_features: usize,
        options: CombatDenseOptions,
        parallel: ParallelPlan,
    ) -> Vec<f64> {
        let levels = BatchLevels::from_ids(batch, n_samples).unwrap();
        combat_dense_impl_with_parallel(
            CombatDenseInput {
                values,
                n_samples,
                n_features,
                batch,
                covariates: None,
            },
            options,
            levels,
            parallel,
        )
        .unwrap()
        .adjusted
    }

    #[test]
    fn forced_parallel_matches_serial_parametric_output() {
        let n_samples = 32;
        let n_features = 96;
        let values = deterministic_matrix(n_samples, n_features);
        let batch = two_batch_labels(n_samples);
        let options = CombatDenseOptions::default();

        let serial = run_with_plan(
            &values,
            &batch,
            n_samples,
            n_features,
            options,
            ParallelPlan::serial(),
        );
        let parallel = run_with_plan(
            &values,
            &batch,
            n_samples,
            n_features,
            options,
            ParallelPlan::parallel(),
        );

        assert_eq!(parallel, serial);
    }

    #[test]
    fn forced_parallel_matches_serial_nonparametric_output() {
        let n_samples = 32;
        let n_features = 96;
        let values = deterministic_matrix(n_samples, n_features);
        let batch = two_batch_labels(n_samples);
        let options = CombatDenseOptions {
            par_prior: false,
            ..CombatDenseOptions::default()
        };

        let serial = run_with_plan(
            &values,
            &batch,
            n_samples,
            n_features,
            options,
            ParallelPlan::serial(),
        );
        let parallel = run_with_plan(
            &values,
            &batch,
            n_samples,
            n_features,
            options,
            ParallelPlan::parallel(),
        );

        assert_eq!(parallel, serial);
    }
}
