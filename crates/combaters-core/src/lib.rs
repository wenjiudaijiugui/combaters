//! Dense finite ComBat core for biological batch-effect correction.
//!
//! The public matrix contract is row-major `samples x features`:
//! `values[sample * n_features + feature]`.

mod adjust;
mod api;
mod batch;
mod dense;
mod design;
mod error;
mod layout;
mod nonparametric;
mod parametric;
mod standardize;
mod zero_variance;

pub use api::{
    CombatDenseInput, CombatDenseOptions, CombatDenseReport, CombatDenseResult, CovariateMatrix,
    combat_dense,
};
pub use error::CombatError;
