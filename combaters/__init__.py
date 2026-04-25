from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._combaters import combat as _combat


def _float64_matrix(name: str, value: npt.ArrayLike) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 2D float64 array-like") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D float64 array-like")
    return np.ascontiguousarray(array, dtype=np.float64)


def _int64_vector(name: str, value: npt.ArrayLike) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 1D integer array-like") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D integer array-like")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer labels")
    return np.ascontiguousarray(array, dtype=np.int64)


def combat(
    values: npt.ArrayLike,
    batch: npt.ArrayLike,
    mod: npt.ArrayLike | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]:
    """Run dense ComBat after coercing array-like inputs to contiguous arrays."""
    values_array = _float64_matrix("values", values)
    batch_array = _int64_vector("batch", batch)
    mod_array = None if mod is None else _float64_matrix("mod", mod)
    return _combat(values_array, batch_array, mod_array, par_prior, mean_only, ref_batch)


__all__ = ["combat"]
