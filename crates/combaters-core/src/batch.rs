use std::collections::BTreeMap;

use crate::error::CombatError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchLevels {
    pub raw_levels: Vec<usize>,
    pub sample_to_level: Vec<usize>,
    pub counts: Vec<usize>,
}

impl BatchLevels {
    pub(crate) fn from_ids(batch: &[usize], n_samples: usize) -> Result<Self, CombatError> {
        if batch.len() != n_samples {
            return Err(CombatError::BatchLengthMismatch {
                n_samples,
                batch_len: batch.len(),
            });
        }

        let mut unique = BTreeMap::new();
        for &raw in batch {
            unique.entry(raw).or_insert(());
        }

        if unique.len() < 2 {
            return Err(CombatError::NeedAtLeastTwoBatches);
        }

        let raw_levels: Vec<usize> = unique.keys().copied().collect();
        let raw_to_level: BTreeMap<usize, usize> = raw_levels
            .iter()
            .copied()
            .enumerate()
            .map(|(level, raw)| (raw, level))
            .collect();

        let mut sample_to_level = Vec::with_capacity(batch.len());
        let mut counts = vec![0usize; raw_levels.len()];
        for &raw in batch {
            let level = *raw_to_level
                .get(&raw)
                .expect("batch level was inserted before lookup");
            sample_to_level.push(level);
            counts[level] += 1;
        }

        Ok(Self {
            raw_levels,
            sample_to_level,
            counts,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.raw_levels.len()
    }

    pub(crate) fn resolve_raw(&self, raw: usize) -> Option<usize> {
        self.raw_levels.binary_search(&raw).ok()
    }

    pub(crate) fn singleton_raw_ids(&self) -> Vec<usize> {
        self.raw_levels
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .filter_map(|(raw, count)| (count == 1).then_some(raw))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::BatchLevels;

    #[test]
    fn compacts_sparse_ids_in_sorted_raw_order() {
        let levels = BatchLevels::from_ids(&[1000, 42, 1000, 7], 4).unwrap();
        assert_eq!(levels.raw_levels, vec![7, 42, 1000]);
        assert_eq!(levels.sample_to_level, vec![2, 1, 2, 0]);
        assert_eq!(levels.counts, vec![1, 1, 2]);
    }
}
