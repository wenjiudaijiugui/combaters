use std::path::Path;

pub fn read_csv_matrix(path: &Path) -> Vec<Vec<f64>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .expect("fixture CSV opens");

    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.expect("fixture CSV row parses");
        let row = record
            .iter()
            .map(|field| field.parse::<f64>().expect("fixture value parses as f64"))
            .collect::<Vec<_>>();
        rows.push(row);
    }
    rows
}

pub fn read_batch(path: &Path) -> Vec<usize> {
    let text = std::fs::read_to_string(path).expect("batch fixture opens");
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().parse::<usize>().expect("batch id parses"))
        .collect()
}

pub fn flatten_rows(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

pub fn assert_matrix_close(actual: &[f64], expected: &[f64], abs_tol: f64, rel_tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (left - right).abs();
        let scale = left.abs().max(right.abs());
        assert!(
            diff <= abs_tol || diff <= rel_tol * scale,
            "matrix mismatch at flat index {idx}: actual={left}, expected={right}, diff={diff}, scale={scale}"
        );
    }
}
