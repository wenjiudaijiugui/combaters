use combaters_core::{
    CombatDenseInput, CombatDenseOptions, CombatError, CovariateMatrix, combat_dense,
};

fn options() -> CombatDenseOptions {
    CombatDenseOptions::default()
}

#[test]
fn rejects_value_shape_mismatch() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    let batch = [0, 1, 0];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 3,
        n_features: 2,
        batch: &batch,
        covariates: None,
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(
        err,
        CombatError::ShapeMismatch {
            expected: 6,
            actual: 5,
            context: "values"
        }
    ));
}

#[test]
fn rejects_shape_overflow() {
    let values = [];
    let batch = [];
    let input = CombatDenseInput {
        values: &values,
        n_samples: usize::MAX,
        n_features: 2,
        batch: &batch,
        covariates: None,
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(
        err,
        CombatError::ShapeOverflow { context: "values" }
    ));
}

#[test]
fn rejects_zero_samples() {
    let values = [];
    let batch = [];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 0,
        n_features: 2,
        batch: &batch,
        covariates: None,
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(
        err,
        CombatError::EmptyInput {
            context: "n_samples"
        }
    ));
}

#[test]
fn rejects_zero_features() {
    let values = [];
    let batch = [0, 1];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 2,
        n_features: 0,
        batch: &batch,
        covariates: None,
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(
        err,
        CombatError::EmptyInput {
            context: "n_features"
        }
    ));
}

#[test]
fn rejects_batch_length_mismatch() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let batch = [0, 1];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 3,
        n_features: 2,
        batch: &batch,
        covariates: None,
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(
        err,
        CombatError::BatchLengthMismatch {
            n_samples: 3,
            batch_len: 2
        }
    ));
}

#[test]
fn rejects_one_batch() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let batch = [7, 7, 7];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 3,
        n_features: 2,
        batch: &batch,
        covariates: None,
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(err, CombatError::NeedAtLeastTwoBatches));
}

#[test]
fn rejects_non_finite_dense_value_with_coordinates() {
    let values = [1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0];
    let batch = [0, 1, 0];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 3,
        n_features: 2,
        batch: &batch,
        covariates: None,
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(
        err,
        CombatError::NonFiniteValue {
            sample: 1,
            feature: 1
        }
    ));
}

#[test]
fn rejects_covariate_shape_mismatch() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let batch = [0, 1, 0];
    let covariates = [0.1, 0.2];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 3,
        n_features: 2,
        batch: &batch,
        covariates: Some(CovariateMatrix {
            values: &covariates,
            n_covariates: 1,
        }),
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(
        err,
        CombatError::CovariateShapeMismatch {
            n_samples: 3,
            n_covariates: 1,
            len: 2
        }
    ));
}

#[test]
fn rejects_non_finite_covariate_value_with_coordinates() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let batch = [0, 1, 0];
    let covariates = [0.1, f64::INFINITY, 0.3];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 3,
        n_features: 2,
        batch: &batch,
        covariates: Some(CovariateMatrix {
            values: &covariates,
            n_covariates: 1,
        }),
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(
        err,
        CombatError::NonFiniteDesignValue {
            sample: 1,
            column: 0,
            context: "covariates"
        }
    ));
}

#[test]
fn accepts_covariate_intercept_column_by_dropping_it() {
    let values = [
        4.0, 1.0, 7.0, 2.0, 5.0, 1.5, 8.0, 2.5, 6.5, 2.0, 8.8, 3.0, 11.0, 7.0, 2.0, 6.0, 12.5, 7.5,
        2.5, 6.5, 13.0, 8.0, 3.0, 7.0,
    ];
    let batch = [10, 10, 10, 20, 20, 20];
    let covariates = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
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

    let result = combat_dense(input, options()).unwrap();
    assert_eq!(result.n_samples, 6);
    assert_eq!(result.n_features, 4);
}

#[test]
fn rejects_duplicate_covariate_columns() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let batch = [0, 1, 0, 1];
    let covariates = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 4,
        n_features: 2,
        batch: &batch,
        covariates: Some(CovariateMatrix {
            values: &covariates,
            n_covariates: 2,
        }),
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(err, CombatError::InvalidDesign { reason } if reason.contains("duplicate")));
}

#[test]
fn rejects_covariate_column_collinear_with_batch_indicators() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let batch = [0, 0, 1, 1];
    let covariates = [5.0, 5.0, 9.0, 9.0];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 4,
        n_features: 2,
        batch: &batch,
        covariates: Some(CovariateMatrix {
            values: &covariates,
            n_covariates: 1,
        }),
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(err, CombatError::InvalidDesign { reason } if reason.contains("batch")));
}

#[test]
fn rejects_too_many_design_columns_before_fitting() {
    let values = [1.0, 2.0, 3.0];
    let batch = [0, 1, 0];
    let covariates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 3,
        n_features: 1,
        batch: &batch,
        covariates: Some(CovariateMatrix {
            values: &covariates,
            n_covariates: 2,
        }),
    };

    let err = combat_dense(input, options()).unwrap_err();
    assert!(matches!(err, CombatError::InvalidDesign { reason } if reason.contains("too many")));
}

#[test]
fn accepts_valid_covariates() {
    let values = [
        4.0, 1.0, 7.0, 2.0, 5.0, 1.5, 8.0, 2.5, 6.5, 2.0, 8.8, 3.0, 11.0, 7.0, 2.0, 6.0, 12.5, 7.5,
        2.5, 6.5, 13.0, 8.0, 3.0, 7.0,
    ];
    let batch = [10, 10, 10, 20, 20, 20];
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

    let result = combat_dense(input, options()).unwrap();
    assert_eq!(result.adjusted.len(), values.len());
}

#[test]
fn rejects_missing_reference_batch_by_raw_id() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let batch = [10, 20, 10];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 3,
        n_features: 2,
        batch: &batch,
        covariates: None,
    };
    let mut options = options();
    options.ref_batch = Some(30);

    let err = combat_dense(input, options).unwrap_err();
    assert!(matches!(
        err,
        CombatError::MissingReferenceBatch { requested: 30 }
    ));
}

#[test]
fn accepts_sparse_batch_ids() {
    let values = [
        4.0, 1.0, 7.0, 2.0, 5.0, 1.5, 8.0, 2.5, 6.5, 2.0, 8.8, 3.0, 11.0, 7.0, 2.0, 6.0, 12.5, 7.5,
        2.5, 6.5, 13.0, 8.0, 3.0, 7.0,
    ];
    let batch = [42, 42, 42, 1000, 1000, 1000];
    let input = CombatDenseInput {
        values: &values,
        n_samples: 6,
        n_features: 4,
        batch: &batch,
        covariates: None,
    };

    let result = combat_dense(input, options()).unwrap();
    assert_eq!(result.n_samples, 6);
    assert_eq!(result.n_features, 4);
    assert_eq!(result.adjusted.len(), values.len());
}
