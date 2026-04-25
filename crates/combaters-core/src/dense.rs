use crate::adjust::adjust_parametric;
use crate::api::{CombatDenseInput, CombatDenseOptions, CombatDenseReport, CombatDenseResult};
use crate::batch::BatchLevels;
use crate::error::CombatError;
use crate::layout::row_major_index;
use crate::nonparametric::fit_nonparametric;
use crate::parametric::{fit_parametric, fit_unshrunken_mean_only};
use crate::standardize::standardize;
use crate::zero_variance::{project_features, reinsert_features, select_nonzero_variance_features};

pub(crate) fn combat_dense_impl(
    input: CombatDenseInput<'_>,
    options: CombatDenseOptions,
    levels: BatchLevels,
) -> Result<CombatDenseResult, CombatError> {
    let singleton_batches = levels.singleton_raw_ids();
    let ref_level = options.ref_batch.and_then(|raw| levels.resolve_raw(raw));

    let selection =
        select_nonzero_variance_features(input.values, input.n_samples, input.n_features, &levels);
    let fitting_features = selection.kept_features.len();
    let underpowered_for_eb = fitting_features < 2;
    let effective_mean_only =
        options.mean_only || !singleton_batches.is_empty() || underpowered_for_eb;
    if selection.kept_features.is_empty() {
        return Ok(CombatDenseResult {
            adjusted: input.values.to_vec(),
            n_samples: input.n_samples,
            n_features: input.n_features,
            report: CombatDenseReport {
                effective_mean_only,
                singleton_batches,
                zero_variance_features: selection.zero_variance_features,
            },
        });
    }

    let fitting_values = project_features(
        input.values,
        input.n_samples,
        input.n_features,
        &selection.kept_features,
    );
    let state = standardize(
        &fitting_values,
        input.n_samples,
        fitting_features,
        &levels,
        input.covariates,
        ref_level,
    )?;
    let estimates = if underpowered_for_eb {
        fit_unshrunken_mean_only(&state, &levels, ref_level)?
    } else if options.par_prior {
        fit_parametric(&state, &levels, effective_mean_only, ref_level)?
    } else {
        fit_nonparametric(&state, &levels, effective_mean_only, ref_level)?
    };
    let adjusted_kept = adjust_parametric(&state, &estimates, &levels);
    let mut adjusted = reinsert_features(
        input.values,
        &adjusted_kept,
        input.n_samples,
        input.n_features,
        &selection.kept_features,
    );
    if let Some(ref_level) = ref_level {
        preserve_reference_batch(
            input.values,
            &mut adjusted,
            input.n_samples,
            input.n_features,
            &levels,
            ref_level,
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
) {
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
