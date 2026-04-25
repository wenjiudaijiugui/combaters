from __future__ import annotations

import numpy as np
import pytest


def balanced_matrix() -> np.ndarray:
    values = [
        4.0,
        1.0,
        7.0,
        2.0,
        5.0,
        1.5,
        8.0,
        2.5,
        6.5,
        2.0,
        8.8,
        3.0,
        11.0,
        7.0,
        2.0,
        6.0,
        12.5,
        7.5,
        2.5,
        6.5,
        13.0,
        8.0,
        3.0,
        7.0,
    ]
    return np.asarray(values, dtype=np.float64).reshape((6, 4))


def test_combat_returns_shape_and_report() -> None:
    from combaters import combat

    batch = np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64)
    result = combat(balanced_matrix(), batch)

    assert result["n_samples"] == 6
    assert result["n_features"] == 4
    adjusted = result["adjusted"]
    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == (6, 4)
    assert adjusted.dtype == np.float64
    assert result["report"] == {
        "effective_mean_only": False,
        "singleton_batches": [],
        "zero_variance_features": [],
    }


def test_combat_preserves_samples_x_features_shape_contract() -> None:
    from combaters import combat

    with pytest.raises(ValueError, match="batch length mismatch"):
        combat(
            balanced_matrix()[:-1, :],
            np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64),
        )
