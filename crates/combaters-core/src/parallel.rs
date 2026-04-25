use std::env;

pub(crate) const AUTO_PARALLEL_MIN_CELLS: usize = 65_536;
pub(crate) const AUTO_PARALLEL_MIN_JOBS: usize = 64;
const PARALLEL_ENV: &str = "COMBATERS_PARALLEL";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParallelMode {
    Serial,
    Parallel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParallelPlan {
    mode: ParallelMode,
}

impl ParallelPlan {
    #[cfg(test)]
    pub(crate) fn serial() -> Self {
        Self {
            mode: ParallelMode::Serial,
        }
    }

    #[cfg(test)]
    pub(crate) fn parallel() -> Self {
        Self {
            mode: ParallelMode::Parallel,
        }
    }

    pub(crate) fn for_shape(n_samples: usize, n_features: usize, n_levels: usize) -> Self {
        let mode = env::var(PARALLEL_ENV)
            .ok()
            .and_then(|value| parse_override(&value))
            .unwrap_or_else(|| automatic_mode(n_samples, n_features, n_levels));
        Self { mode }
    }

    pub(crate) fn should_parallelize(self, jobs: usize) -> bool {
        self.mode == ParallelMode::Parallel && jobs > 1
    }
}

fn automatic_mode(n_samples: usize, n_features: usize, n_levels: usize) -> ParallelMode {
    let cells = n_samples.saturating_mul(n_features);
    let jobs = n_features.saturating_mul(n_levels.max(1));
    if cells >= AUTO_PARALLEL_MIN_CELLS && jobs >= AUTO_PARALLEL_MIN_JOBS {
        ParallelMode::Parallel
    } else {
        ParallelMode::Serial
    }
}

fn parse_override(value: &str) -> Option<ParallelMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "false" | "off" | "serial" => Some(ParallelMode::Serial),
        "1" | "true" | "on" | "parallel" => Some(ParallelMode::Parallel),
        "auto" | "" => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AUTO_PARALLEL_MIN_CELLS, AUTO_PARALLEL_MIN_JOBS, ParallelMode, automatic_mode,
        parse_override,
    };

    #[test]
    fn automatic_mode_keeps_small_data_serial() {
        assert_eq!(automatic_mode(6, 4, 2), ParallelMode::Serial);
    }

    #[test]
    fn automatic_mode_enables_large_independent_work() {
        assert_eq!(
            automatic_mode(
                AUTO_PARALLEL_MIN_CELLS / AUTO_PARALLEL_MIN_JOBS,
                AUTO_PARALLEL_MIN_JOBS,
                2
            ),
            ParallelMode::Parallel
        );
    }

    #[test]
    fn env_override_accepts_only_narrow_values() {
        assert_eq!(parse_override("off"), Some(ParallelMode::Serial));
        assert_eq!(parse_override("parallel"), Some(ParallelMode::Parallel));
        assert_eq!(parse_override("auto"), None);
        assert_eq!(parse_override("not-a-mode"), None);
    }
}
