from __future__ import annotations

import numpy as np
import pytest


def balanced_values() -> list[float]:
    return [
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


def balanced_matrix() -> np.ndarray:
    return np.asarray(balanced_values(), dtype=np.float64).reshape((6, 4))


def balanced_batch() -> np.ndarray:
    return np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64)


def test_combat_returns_numpy_adjusted_matrix() -> None:
    from combaters import combat

    matrix = balanced_matrix()
    batch = balanced_batch()

    result = combat(matrix, batch)

    adjusted = result["adjusted"]
    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == (6, 4)
    assert adjusted.dtype == np.float64
    assert result["n_samples"] == 6
    assert result["n_features"] == 4
    assert result["report"] == {
        "effective_mean_only": False,
        "singleton_batches": [],
        "zero_variance_features": [],
    }


def test_combat_accepts_mod_matrix() -> None:
    from combaters import combat

    mod = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64).reshape((6, 1))
    result = combat(balanced_matrix(), balanced_batch(), mod=mod)

    adjusted = result["adjusted"]
    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == (6, 4)


def test_combat_par_prior_false_returns_nonparametric_result() -> None:
    from combaters import combat

    matrix = balanced_matrix()
    batch = balanced_batch()

    parametric = combat(matrix, batch)
    nonparametric = combat(matrix, batch, par_prior=False)

    assert nonparametric["adjusted"].shape == matrix.shape
    assert np.all(np.isfinite(nonparametric["adjusted"]))
    assert nonparametric["report"]["effective_mean_only"] is False
    assert np.any(np.abs(nonparametric["adjusted"] - parametric["adjusted"]) > 1e-8)


def test_combat_par_prior_false_supports_mod_and_ref_batch() -> None:
    from combaters import combat

    matrix = balanced_matrix()
    batch = balanced_batch()
    mod = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64).reshape((6, 1))

    result = combat(matrix, batch, mod=mod, par_prior=False, ref_batch=20)

    np.testing.assert_array_equal(result["adjusted"][3:], matrix[3:])
    assert np.all(np.isfinite(result["adjusted"]))
    assert result["report"]["effective_mean_only"] is False


def test_combat_mean_only_preserves_within_batch_scale() -> None:
    from combaters import combat

    matrix = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [3.0, 14.0, 103.0],
            [5.0, 20.0, 109.0],
            [7.0, 24.0, 112.0],
        ],
        dtype=np.float64,
    )
    batch = np.asarray([10, 10, 20, 20], dtype=np.int64)

    result = combat(matrix, batch, mean_only=True)
    adjusted = result["adjusted"]

    np.testing.assert_allclose(adjusted[0] - adjusted[1], matrix[0] - matrix[1])
    np.testing.assert_allclose(adjusted[2] - adjusted[3], matrix[2] - matrix[3])
    assert result["report"]["effective_mean_only"] is True


def test_combat_par_prior_false_mean_only_preserves_within_batch_scale() -> None:
    from combaters import combat

    matrix = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [3.0, 14.0, 103.0],
            [5.0, 20.0, 109.0],
            [7.0, 24.0, 112.0],
        ],
        dtype=np.float64,
    )
    batch = np.asarray([10, 10, 20, 20], dtype=np.int64)

    result = combat(matrix, batch, par_prior=False, mean_only=True)
    adjusted = result["adjusted"]

    np.testing.assert_allclose(adjusted[0] - adjusted[1], matrix[0] - matrix[1])
    np.testing.assert_allclose(adjusted[2] - adjusted[3], matrix[2] - matrix[3])
    assert result["report"]["effective_mean_only"] is True


def test_combat_ref_batch_preserves_reference_rows() -> None:
    from combaters import combat

    matrix = balanced_matrix()
    batch = balanced_batch()

    result = combat(matrix, batch, ref_batch=20)
    adjusted = result["adjusted"]

    np.testing.assert_array_equal(adjusted[3:], matrix[3:])
    assert np.any(np.abs(adjusted[:3] - matrix[:3]) > 1e-8)
    assert result["report"]["effective_mean_only"] is False


def test_combat_singleton_batch_uses_effective_mean_only() -> None:
    from combaters import combat

    matrix = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [3.0, 14.0, 103.0],
            [5.0, 20.0, 109.0],
            [7.0, 24.0, 112.0],
            [9.0, 30.0, 120.0],
        ],
        dtype=np.float64,
    )
    batch = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)

    result = combat(matrix, batch)

    assert result["adjusted"].shape == matrix.shape
    assert result["report"]["effective_mean_only"] is True
    assert result["report"]["singleton_batches"] == [2]


def test_combat_all_zero_variance_features_returns_original_matrix() -> None:
    from combaters import combat

    matrix = np.asarray(
        [
            [1.0, 5.0],
            [1.0, 5.0],
            [3.0, 7.0],
            [3.0, 7.0],
        ],
        dtype=np.float64,
    )
    batch = np.asarray([10, 10, 20, 20], dtype=np.int64)

    result = combat(matrix, batch)

    np.testing.assert_array_equal(result["adjusted"], matrix)
    assert result["report"]["zero_variance_features"] == [0, 1]
    assert result["report"]["effective_mean_only"] is True


def test_combat_single_feature_skips_eb_and_uses_mean_only_adjustment() -> None:
    from combaters import combat

    matrix = np.asarray([[1.0], [3.0], [7.0], [9.0]], dtype=np.float64)
    batch = np.asarray([10, 10, 20, 20], dtype=np.int64)

    result = combat(matrix, batch)
    nonparametric_result = combat(matrix, batch, par_prior=False)

    for result in [result, nonparametric_result]:
        np.testing.assert_allclose(result["adjusted"], np.asarray([[4.0], [6.0], [4.0], [6.0]]))
        assert result["report"]["zero_variance_features"] == []
        assert result["report"]["effective_mean_only"] is True


def test_combat_rejects_mod_row_mismatch() -> None:
    from combaters import combat

    mod = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float64).reshape((5, 1))

    with pytest.raises(ValueError, match="mod row count"):
        combat(balanced_matrix(), balanced_batch(), mod=mod)


def test_combat_rejects_fortran_order_values() -> None:
    from combaters import combat

    matrix = np.asfortranarray(balanced_matrix())

    with pytest.raises(ValueError, match="C-contiguous row-major"):
        combat(matrix, balanced_batch())


def test_combat_rejects_strided_batch() -> None:
    from combaters import combat

    batch = np.asarray([10, 99, 10, 99, 10, 99, 20, 99, 20, 99, 20, 99], dtype=np.int64)[::2]

    with pytest.raises(ValueError, match="contiguous int64"):
        combat(balanced_matrix(), batch)


def test_combat_rejects_negative_batch_id() -> None:
    from combaters import combat

    batch = balanced_batch()
    batch[1] = -1

    with pytest.raises(ValueError, match="non-negative"):
        combat(balanced_matrix(), batch)
