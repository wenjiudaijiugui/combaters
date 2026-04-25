use crate::batch::BatchLevels;
use crate::layout::row_major_index;
use crate::parallel::ParallelPlan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FeatureSelection {
    pub kept_features: Vec<usize>,
    pub zero_variance_features: Vec<usize>,
}

pub(crate) fn select_nonzero_variance_features(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    levels: &BatchLevels,
    parallel: ParallelPlan,
) -> FeatureSelection {
    let zero_variance: Vec<bool> = if parallel.should_parallelize(n_features) {
        use rayon::prelude::*;

        (0..n_features)
            .into_par_iter()
            .map(|feature| {
                feature_has_zero_variance(values, n_samples, n_features, levels, feature)
            })
            .collect()
    } else {
        (0..n_features)
            .map(|feature| {
                feature_has_zero_variance(values, n_samples, n_features, levels, feature)
            })
            .collect()
    };

    let mut kept_features = Vec::new();
    let mut zero_variance_features = Vec::new();
    for (feature, is_zero) in zero_variance.into_iter().enumerate() {
        if is_zero {
            zero_variance_features.push(feature);
        } else {
            kept_features.push(feature);
        }
    }

    FeatureSelection {
        kept_features,
        zero_variance_features,
    }
}

fn feature_has_zero_variance(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    levels: &BatchLevels,
    feature: usize,
) -> bool {
    for level in 0..levels.len() {
        if levels.counts[level] <= 1 {
            continue;
        }

        let mut first = None;
        let mut all_equal = true;
        for sample in 0..n_samples {
            if levels.sample_to_level[sample] != level {
                continue;
            }

            let value = values[row_major_index(sample, feature, n_features)];
            match first {
                Some(first_value) if value != first_value => {
                    all_equal = false;
                    break;
                }
                Some(_) => {}
                None => first = Some(value),
            }
        }

        if all_equal {
            return true;
        }
    }

    false
}

pub(crate) fn project_features(
    values: &[f64],
    n_samples: usize,
    n_features: usize,
    kept_features: &[usize],
    parallel: ParallelPlan,
) -> Vec<f64> {
    let kept_len = kept_features.len();
    let mut projected = vec![0.0; n_samples * kept_len];
    if parallel.should_parallelize(n_samples) {
        use rayon::prelude::*;

        projected
            .par_chunks_mut(kept_len)
            .enumerate()
            .for_each(|(sample, row)| {
                for (kept_idx, &feature) in kept_features.iter().enumerate() {
                    row[kept_idx] = values[row_major_index(sample, feature, n_features)];
                }
            });
    } else {
        for sample in 0..n_samples {
            let dst_start = sample * kept_len;
            for (kept_idx, &feature) in kept_features.iter().enumerate() {
                projected[dst_start + kept_idx] =
                    values[row_major_index(sample, feature, n_features)];
            }
        }
    }
    projected
}

pub(crate) fn reinsert_features(
    original: &[f64],
    adjusted_kept: &[f64],
    n_samples: usize,
    n_features: usize,
    kept_features: &[usize],
    parallel: ParallelPlan,
) -> Vec<f64> {
    let mut adjusted = original.to_vec();
    let kept_len = kept_features.len();
    if parallel.should_parallelize(n_samples) {
        use rayon::prelude::*;

        adjusted
            .par_chunks_mut(n_features)
            .enumerate()
            .for_each(|(sample, row)| {
                for (kept_idx, &feature) in kept_features.iter().enumerate() {
                    let src = row_major_index(sample, kept_idx, kept_len);
                    row[feature] = adjusted_kept[src];
                }
            });
    } else {
        for sample in 0..n_samples {
            for (kept_idx, &feature) in kept_features.iter().enumerate() {
                let dst = row_major_index(sample, feature, n_features);
                let src = row_major_index(sample, kept_idx, kept_len);
                adjusted[dst] = adjusted_kept[src];
            }
        }
    }
    adjusted
}

#[cfg(test)]
mod tests {
    use crate::batch::BatchLevels;
    use crate::parallel::ParallelPlan;

    use super::{project_features, reinsert_features, select_nonzero_variance_features};

    #[test]
    fn detects_features_constant_within_any_multi_sample_batch() {
        let values = [
            1.0, 5.0, 7.0, 2.0, 5.0, 8.0, 10.0, 6.0, 9.0, 11.0, 6.0, 10.0,
        ];
        let batch = [0, 0, 1, 1];
        let levels = BatchLevels::from_ids(&batch, 4).unwrap();

        let selection =
            select_nonzero_variance_features(&values, 4, 3, &levels, ParallelPlan::serial());

        assert_eq!(selection.kept_features, vec![0, 2]);
        assert_eq!(selection.zero_variance_features, vec![1]);
    }

    #[test]
    fn projects_and_reinserts_kept_features_in_original_row_major_order() {
        let values = [1.0, 5.0, 7.0, 2.0, 5.0, 8.0];
        let projected = project_features(&values, 2, 3, &[0, 2], ParallelPlan::serial());
        assert_eq!(projected, vec![1.0, 7.0, 2.0, 8.0]);

        let adjusted = reinsert_features(
            &values,
            &[10.0, 70.0, 20.0, 80.0],
            2,
            3,
            &[0, 2],
            ParallelPlan::serial(),
        );
        assert_eq!(adjusted, vec![10.0, 5.0, 70.0, 20.0, 5.0, 80.0]);
    }
}
