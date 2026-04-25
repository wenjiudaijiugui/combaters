from __future__ import annotations

import operator
from typing import Any

import numpy as np
import numpy.typing as npt

from ._combaters import combat as _combat

_NO_INT_REF = object()


def _float64_matrix(name: str, value: npt.ArrayLike) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 2D float64 array-like") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D float64 array-like")
    return np.ascontiguousarray(array, dtype=np.float64)


def combat(
    values: npt.ArrayLike,
    batch: npt.ArrayLike,
    mod: npt.ArrayLike | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: object | None = None,
) -> dict[str, object]:
    """Run dense ComBat after coercing array-like inputs to contiguous arrays."""
    values_array = _float64_matrix("values", values)
    batch_array, ref_arg, levels = _prepare_batch(batch, ref_batch)
    mod_array = None if mod is None else _float64_matrix("mod", mod)
    result = _combat(values_array, batch_array, mod_array, par_prior, mean_only, ref_arg)
    if levels is not None:
        _restore_report_labels(result, levels)
    return result


def _prepare_batch(
    batch: object,
    ref_batch: object | None,
) -> tuple[np.ndarray, int | None, list[object] | None]:
    try:
        batch_array = np.asarray(batch)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch must be a 1D array-like vector") from exc
    if batch_array.ndim != 1:
        raise ValueError("batch must be a 1D array-like vector")

    fast_ref = _nonnegative_int_ref(ref_batch)
    if np.issubdtype(batch_array.dtype, np.integer) and not bool(np.any(batch_array < 0)):
        if ref_batch is None or fast_ref is not _NO_INT_REF:
            return (
                np.ascontiguousarray(batch_array, dtype=np.int64),
                None if ref_batch is None else fast_ref,
                None,
            )
        raise ValueError(f"missing reference batch: {ref_batch!r}")

    labels = np.asarray(batch, dtype=object)
    compact, levels = _factorize_labels(labels)
    if ref_batch is None:
        return compact, None, levels

    ref_level = _find_label(levels, ref_batch)
    if ref_level is None:
        raise ValueError(f"missing reference batch: {ref_batch!r}")
    return compact, ref_level, levels


def _nonnegative_int_ref(ref_batch: object | None) -> int | object:
    if ref_batch is None:
        return _NO_INT_REF
    try:
        ref = operator.index(ref_batch)
    except TypeError:
        return _NO_INT_REF
    if ref < 0:
        return _NO_INT_REF
    return ref


def _factorize_labels(labels: np.ndarray) -> tuple[np.ndarray, list[object]]:
    compact = np.empty(labels.shape[0], dtype=np.int64)
    levels: list[object] = []
    for sample, label in enumerate(labels.tolist()):
        level = _find_label(levels, label)
        if level is None:
            level = len(levels)
            levels.append(label)
        compact[sample] = level
    return compact, levels


def _find_label(levels: list[object], label: object) -> int | None:
    for index, level in enumerate(levels):
        if _labels_equal(level, label):
            return index
    return None


def _labels_equal(left: object, right: object) -> bool:
    try:
        equal = left == right
    except Exception:
        return False
    if isinstance(equal, np.ndarray):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _restore_report_labels(result: dict[str, Any], levels: list[object]) -> None:
    report = result.get("report")
    if not isinstance(report, dict):
        return

    singleton_batches = report.get("singleton_batches")
    if singleton_batches is None:
        return
    report["singleton_batches"] = [levels[int(level)] for level in singleton_batches]


__all__ = ["combat"]
