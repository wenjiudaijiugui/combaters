use combaters_core::{CombatDenseInput, CombatDenseOptions, CovariateMatrix, combat_dense};

fn balanced_values() -> Vec<f64> {
    vec![
        4.0, 1.0, 7.0, 2.0, 5.0, 1.5, 8.0, 2.5, 6.5, 2.0, 8.8, 3.0, 11.0, 7.0, 2.0, 6.0, 12.5, 7.5,
        2.5, 6.5, 13.0, 8.0, 3.0, 7.0,
    ]
}

fn balanced_batch() -> Vec<usize> {
    vec![10, 10, 10, 20, 20, 20]
}

fn assert_only_missing_positions_are_nan(values: &[f64], missing_positions: &[usize]) {
    for (idx, value) in values.iter().enumerate() {
        if missing_positions.contains(&idx) {
            assert!(
                value.is_nan(),
                "expected missing position {idx} to remain NaN"
            );
        } else {
            assert!(
                value.is_finite(),
                "expected adjusted value {idx} to be finite"
            );
        }
    }
}

#[test]
fn balanced_two_batch_parametric_returns_finite_same_shape_output() {
    let values = balanced_values();
    let original = values.clone();
    let batch = balanced_batch();
    let input = CombatDenseInput {
        values: &values,
        n_samples: 6,
        n_features: 4,
        batch: &batch,
        covariates: None,
    };

    let result = combat_dense(input, CombatDenseOptions::default()).unwrap();

    assert_eq!(result.n_samples, 6);
    assert_eq!(result.n_features, 4);
    assert_eq!(result.adjusted.len(), values.len());
    assert!(result.adjusted.iter().all(|value| value.is_finite()));
    assert_eq!(values, original);
    assert!(!result.report.effective_mean_only);
    assert!(result.report.singleton_batches.is_empty());
}

#[test]
fn parametric_adjustment_accepts_missing_values_and_preserves_them() {
    let mut values = balanced_values();
    values[6] = f64::NAN;
    values[16] = f64::NAN;
    let batch = balanced_batch();
    let input = CombatDenseInput {
        values: &values,
        n_samples: 6,
        n_features: 4,
        batch: &batch,
        covariates: None,
    };

    let result = combat_dense(input, CombatDenseOptions::default()).unwrap();

    assert_eq!(result.n_samples, 6);
    assert_eq!(result.n_features, 4);
    assert_only_missing_positions_are_nan(&result.adjusted, &[6, 16]);
}

#[test]
fn nonparametric_adjustment_accepts_missing_values_and_preserves_them() {
    let mut values = balanced_values();
    values[6] = f64::NAN;
    values[16] = f64::NAN;
    let batch = balanced_batch();
    let input = CombatDenseInput {
        values: &values,
        n_samples: 6,
        n_features: 4,
        batch: &batch,
        covariates: None,
    };
    let options = CombatDenseOptions {
        par_prior: false,
        ..CombatDenseOptions::default()
    };

    let result = combat_dense(input, options).unwrap();

    assert_eq!(result.n_samples, 6);
    assert_eq!(result.n_features, 4);
    assert_only_missing_positions_are_nan(&result.adjusted, &[6, 16]);
}

#[test]
fn sparse_batch_ids_match_compact_equivalent_ids() {
    let values = balanced_values();
    let sparse_batch = balanced_batch();
    let compact_batch = vec![0, 0, 0, 1, 1, 1];

    let sparse_result = combat_dense(
        CombatDenseInput {
            values: &values,
            n_samples: 6,
            n_features: 4,
            batch: &sparse_batch,
            covariates: None,
        },
        CombatDenseOptions::default(),
    )
    .unwrap();

    let compact_result = combat_dense(
        CombatDenseInput {
            values: &values,
            n_samples: 6,
            n_features: 4,
            batch: &compact_batch,
            covariates: None,
        },
        CombatDenseOptions::default(),
    )
    .unwrap();

    assert_eq!(sparse_result.adjusted.len(), compact_result.adjusted.len());
    for (left, right) in sparse_result
        .adjusted
        .iter()
        .zip(compact_result.adjusted.iter())
    {
        assert!((left - right).abs() <= 1e-10, "left={left}, right={right}");
    }
}

#[test]
fn par_prior_false_returns_finite_nonparametric_result() {
    let values = balanced_values();
    let batch = balanced_batch();
    let parametric = combat_dense(
        CombatDenseInput {
            values: &values,
            n_samples: 6,
            n_features: 4,
            batch: &batch,
            covariates: None,
        },
        CombatDenseOptions::default(),
    )
    .unwrap();
    let nonparametric = combat_dense(
        CombatDenseInput {
            values: &values,
            n_samples: 6,
            n_features: 4,
            batch: &batch,
            covariates: None,
        },
        CombatDenseOptions {
            par_prior: false,
            ..CombatDenseOptions::default()
        },
    )
    .unwrap();

    assert_eq!(nonparametric.n_samples, 6);
    assert_eq!(nonparametric.n_features, 4);
    assert!(nonparametric.adjusted.iter().all(|value| value.is_finite()));
    assert!(!nonparametric.report.effective_mean_only);
    assert!(
        nonparametric
            .adjusted
            .iter()
            .zip(parametric.adjusted.iter())
            .any(|(left, right)| (left - right).abs() > 1e-8)
    );
}

#[test]
fn reference_batch_is_preserved_by_raw_batch_id() {
    let values = balanced_values();
    let batch = balanced_batch();
    let options = CombatDenseOptions {
        ref_batch: Some(20),
        ..CombatDenseOptions::default()
    };
    let input = CombatDenseInput {
        values: &values,
        n_samples: 6,
        n_features: 4,
        batch: &batch,
        covariates: None,
    };

    let result = combat_dense(input, options).unwrap();

    for sample in 3..6 {
        let start = sample * 4;
        let end = start + 4;
        assert_eq!(&result.adjusted[start..end], &values[start..end]);
    }
    assert!(
        result.adjusted[..12]
            .iter()
            .zip(values[..12].iter())
            .any(|(adjusted, original)| (adjusted - original).abs() > 1e-8)
    );
    assert!(!result.report.effective_mean_only);
}

#[test]
fn valid_covariates_are_supported_in_parametric_slice() {
    let values = balanced_values();
    let batch = balanced_batch();
    let covariates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 6,
        n_features: 4,
        batch: &batch,
        covariates: Some(CovariateMatrix {
            values: &covariates,
            n_covariates: 1,
        }),
    };

    let result = combat_dense(input, CombatDenseOptions::default()).unwrap();
    assert_eq!(result.n_samples, 6);
    assert_eq!(result.n_features, 4);
    assert_eq!(result.adjusted.len(), values.len());
    assert!(result.adjusted.iter().all(|value| value.is_finite()));
}

#[test]
fn nonparametric_supports_covariates_and_preserves_reference_batch() {
    let values = balanced_values();
    let batch = balanced_batch();
    let covariates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 6,
        n_features: 4,
        batch: &batch,
        covariates: Some(CovariateMatrix {
            values: &covariates,
            n_covariates: 1,
        }),
    };
    let options = CombatDenseOptions {
        par_prior: false,
        ref_batch: Some(20),
        ..CombatDenseOptions::default()
    };

    let result = combat_dense(input, options).unwrap();

    for sample in 3..6 {
        let start = sample * 4;
        let end = start + 4;
        assert_eq!(&result.adjusted[start..end], &values[start..end]);
    }
    assert!(result.adjusted.iter().all(|value| value.is_finite()));
    assert!(!result.report.effective_mean_only);
}

#[test]
fn mean_only_adjusts_batch_means_without_scale_adjustment() {
    let values = [
        1.0, 10.0, 100.0, 3.0, 14.0, 103.0, 5.0, 20.0, 109.0, 7.0, 24.0, 112.0,
    ];
    let batch = [10, 10, 20, 20];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 4,
        n_features: 3,
        batch: &batch,
        covariates: None,
    };
    let options = CombatDenseOptions {
        mean_only: true,
        ..CombatDenseOptions::default()
    };

    let result = combat_dense(input, options).unwrap();

    assert!(result.report.effective_mean_only);
    for feature in 0..3 {
        let original_batch_10_diff = values[feature] - values[3 + feature];
        let adjusted_batch_10_diff = result.adjusted[feature] - result.adjusted[3 + feature];
        let original_batch_20_diff = values[6 + feature] - values[9 + feature];
        let adjusted_batch_20_diff = result.adjusted[6 + feature] - result.adjusted[9 + feature];
        assert!((adjusted_batch_10_diff - original_batch_10_diff).abs() <= 1e-10);
        assert!((adjusted_batch_20_diff - original_batch_20_diff).abs() <= 1e-10);
    }
}

#[test]
fn nonparametric_mean_only_adjusts_batch_means_without_scale_adjustment() {
    let values = [
        1.0, 10.0, 100.0, 3.0, 14.0, 103.0, 5.0, 20.0, 109.0, 7.0, 24.0, 112.0,
    ];
    let batch = [10, 10, 20, 20];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 4,
        n_features: 3,
        batch: &batch,
        covariates: None,
    };
    let options = CombatDenseOptions {
        par_prior: false,
        mean_only: true,
        ..CombatDenseOptions::default()
    };

    let result = combat_dense(input, options).unwrap();

    assert!(result.report.effective_mean_only);
    for feature in 0..3 {
        let original_batch_10_diff = values[feature] - values[3 + feature];
        let adjusted_batch_10_diff = result.adjusted[feature] - result.adjusted[3 + feature];
        let original_batch_20_diff = values[6 + feature] - values[9 + feature];
        let adjusted_batch_20_diff = result.adjusted[6 + feature] - result.adjusted[9 + feature];
        assert!((adjusted_batch_10_diff - original_batch_10_diff).abs() <= 1e-10);
        assert!((adjusted_batch_20_diff - original_batch_20_diff).abs() <= 1e-10);
    }
}

#[test]
fn singleton_batch_automatically_uses_effective_mean_only() {
    let values = [
        1.0, 10.0, 100.0, 3.0, 14.0, 103.0, 5.0, 20.0, 109.0, 7.0, 24.0, 112.0, 9.0, 30.0, 120.0,
    ];
    let batch = [0, 0, 1, 1, 2];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 5,
        n_features: 3,
        batch: &batch,
        covariates: None,
    };

    let result = combat_dense(input, CombatDenseOptions::default()).unwrap();

    assert_eq!(result.n_samples, 5);
    assert_eq!(result.n_features, 3);
    assert!(result.adjusted.iter().all(|value| value.is_finite()));
    assert!(result.report.effective_mean_only);
    assert_eq!(result.report.singleton_batches, vec![2]);
}

#[test]
fn nonparametric_singleton_batch_automatically_uses_effective_mean_only() {
    let values = [
        1.0, 10.0, 100.0, 3.0, 14.0, 103.0, 5.0, 20.0, 109.0, 7.0, 24.0, 112.0, 9.0, 30.0, 120.0,
    ];
    let batch = [0, 0, 1, 1, 2];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 5,
        n_features: 3,
        batch: &batch,
        covariates: None,
    };
    let options = CombatDenseOptions {
        par_prior: false,
        ..CombatDenseOptions::default()
    };

    let result = combat_dense(input, options).unwrap();

    assert_eq!(result.n_samples, 5);
    assert_eq!(result.n_features, 3);
    assert!(result.adjusted.iter().all(|value| value.is_finite()));
    assert!(result.report.effective_mean_only);
    assert_eq!(result.report.singleton_batches, vec![2]);
}
