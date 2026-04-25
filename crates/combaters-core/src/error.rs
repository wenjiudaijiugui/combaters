use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CombatError {
    #[error("shape mismatch in {context}: expected {expected}, got {actual}")]
    ShapeMismatch {
        expected: usize,
        actual: usize,
        context: &'static str,
    },
    #[error("shape product overflow in {context}")]
    ShapeOverflow { context: &'static str },
    #[error("empty input for {context}")]
    EmptyInput { context: &'static str },
    #[error("batch length mismatch: n_samples={n_samples}, batch_len={batch_len}")]
    BatchLengthMismatch { n_samples: usize, batch_len: usize },
    #[error(
        "covariate shape mismatch: n_samples={n_samples}, n_covariates={n_covariates}, len={len}"
    )]
    CovariateShapeMismatch {
        n_samples: usize,
        n_covariates: usize,
        len: usize,
    },
    #[error("at least two batches are required")]
    NeedAtLeastTwoBatches,
    #[error("missing reference batch: {requested}")]
    MissingReferenceBatch { requested: usize },
    #[error("non-finite dense value at sample {sample}, feature {feature}")]
    NonFiniteValue { sample: usize, feature: usize },
    #[error("non-finite design value in {context} at sample {sample}, column {column}")]
    NonFiniteDesignValue {
        sample: usize,
        column: usize,
        context: &'static str,
    },
    #[error("singular design matrix")]
    SingularDesign,
    #[error("invalid design: {reason}")]
    InvalidDesign { reason: String },
    #[error("unsupported option: {reason}")]
    UnsupportedOption { reason: String },
    #[error("numerical failure: {reason}")]
    NumericalFailure { reason: String },
}
