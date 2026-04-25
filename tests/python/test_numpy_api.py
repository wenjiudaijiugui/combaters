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


def deterministic_matrix(n_samples: int, n_features: int) -> np.ndarray:
    values = []
    for sample in range(n_samples):
        batch_shift = 0.0 if sample < n_samples // 2 else 3.0
        within_batch = sample % (n_samples // 2)
        for feature in range(n_features):
            signal = feature * 0.03
            jitter = ((sample * 31 + feature * 17) % 11) * 0.007
            values.append(batch_shift + within_batch * 0.19 + signal + jitter)
    return np.asarray(values, dtype=np.float64).reshape((n_samples, n_features))


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


def test_combat_accepts_nan_values_and_preserves_missing_positions() -> None:
    from combaters import combat

    matrix = balanced_matrix()
    matrix[1, 2] = np.nan
    matrix[4, 0] = np.nan

    result = combat(matrix, balanced_batch())
    adjusted = result["adjusted"]

    assert adjusted.shape == matrix.shape
    np.testing.assert_array_equal(np.isnan(adjusted), np.isnan(matrix))
    assert np.all(np.isfinite(adjusted[~np.isnan(adjusted)]))


def test_combat_par_prior_false_accepts_nan_values() -> None:
    from combaters import combat

    matrix = balanced_matrix()
    matrix[1, 2] = np.nan
    matrix[4, 0] = np.nan

    result = combat(matrix, balanced_batch(), par_prior=False)
    adjusted = result["adjusted"]

    assert adjusted.shape == matrix.shape
    np.testing.assert_array_equal(np.isnan(adjusted), np.isnan(matrix))
    assert np.all(np.isfinite(adjusted[~np.isnan(adjusted)]))


def test_combat_accepts_list_values_and_tuple_batch() -> None:
    from combaters import combat

    result = combat(balanced_matrix().tolist(), tuple(balanced_batch().tolist()))
    expected = combat(balanced_matrix(), balanced_batch())

    np.testing.assert_allclose(result["adjusted"], expected["adjusted"])
    assert result["adjusted"].dtype == np.float64


def test_combat_accepts_float32_values() -> None:
    from combaters import combat

    result = combat(balanced_matrix().astype(np.float32), balanced_batch())
    expected = combat(balanced_matrix(), balanced_batch())

    np.testing.assert_allclose(
        result["adjusted"],
        expected["adjusted"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert result["adjusted"].dtype == np.float64


def test_combat_accepts_integer_values() -> None:
    from combaters import combat

    matrix = np.rint(balanced_matrix()).astype(np.int32)

    result = combat(matrix, balanced_batch())

    assert result["adjusted"].shape == matrix.shape
    assert result["adjusted"].dtype == np.float64
    assert np.all(np.isfinite(result["adjusted"]))


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


def test_parallel_override_matches_serial_for_nonparametric(monkeypatch: pytest.MonkeyPatch) -> None:
    from combaters import combat

    matrix = deterministic_matrix(32, 96)
    batch = np.asarray([10] * 16 + [20] * 16, dtype=np.int64)

    monkeypatch.setenv("COMBATERS_PARALLEL", "off")
    serial = combat(matrix, batch, par_prior=False)["adjusted"].copy()
    monkeypatch.setenv("COMBATERS_PARALLEL", "parallel")
    parallel = combat(matrix, batch, par_prior=False)["adjusted"]

    np.testing.assert_array_equal(parallel, serial)


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


def test_combat_accepts_fortran_order_values() -> None:
    from combaters import combat

    matrix = np.asfortranarray(balanced_matrix())
    expected = combat(balanced_matrix(), balanced_batch())

    result = combat(matrix, balanced_batch())

    assert matrix.flags.f_contiguous
    assert not matrix.flags.c_contiguous
    np.testing.assert_allclose(result["adjusted"], expected["adjusted"])


def test_combat_accepts_strided_int32_batch() -> None:
    from combaters import combat

    batch = np.asarray(
        [10, 99, 10, 99, 10, 99, 20, 99, 20, 99, 20, 99],
        dtype=np.int32,
    )[::2]
    expected = combat(balanced_matrix(), balanced_batch())

    result = combat(balanced_matrix(), batch)

    assert batch.dtype == np.int32
    assert not batch.flags.c_contiguous
    np.testing.assert_allclose(result["adjusted"], expected["adjusted"])


def test_combat_rejects_negative_batch_id() -> None:
    from combaters import combat

    batch = balanced_batch()
    batch[1] = -1

    with pytest.raises(ValueError, match="non-negative"):
        combat(balanced_matrix(), batch)


def test_combat_rejects_infinite_values() -> None:
    from combaters import combat

    matrix = balanced_matrix()
    matrix[1, 2] = np.inf

    with pytest.raises(ValueError, match="non-finite dense value"):
        combat(matrix, balanced_batch())
