mod support;

use std::path::{Path, PathBuf};

use combaters_core::{CombatDenseInput, CombatDenseOptions, combat_dense};
use support::{assert_matrix_close, flatten_rows, read_batch, read_csv_matrix};

fn fixture_path(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("oracle")
        .join("fixtures")
        .join("balanced_two_batch_parametric")
        .join(file)
}

#[test]
fn balanced_two_batch_parametric_matches_sva_final_matrix() {
    let input_rows = read_csv_matrix(&fixture_path("input_samples_x_features.csv"));
    let expected_rows = read_csv_matrix(&fixture_path("expected_samples_x_features.csv"));
    let batch = read_batch(&fixture_path("batch.csv"));

    assert_eq!(input_rows.len(), 6);
    assert_eq!(expected_rows.len(), 6);
    assert_eq!(batch.len(), 6);
    assert!(input_rows.iter().all(|row| row.len() == 4));
    assert!(expected_rows.iter().all(|row| row.len() == 4));

    let values = flatten_rows(&input_rows);
    let expected = flatten_rows(&expected_rows);

    let result = combat_dense(
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

    assert_eq!(result.n_samples, 6);
    assert_eq!(result.n_features, 4);
    assert_matrix_close(&result.adjusted, &expected, 1e-8, 1e-10);
}
