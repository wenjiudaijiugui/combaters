from __future__ import annotations

import operator
from typing import Any

import numpy as np

from ._combaters import combat as _combat

_NO_INT_REF = object()


def combat(
    values: object,
    batch: object,
    mod: object | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: object | None = None,
) -> dict[str, object]:
    """Run dense ComBat on a C-contiguous float64 NumPy matrix."""
    batch_arg, ref_arg, levels = _prepare_batch(batch, ref_batch)
    result = _combat(values, batch_arg, mod, par_prior, mean_only, ref_arg)
    if levels is not None:
        _restore_report_labels(result, levels)
    return result


def _prepare_batch(
    batch: object,
    ref_batch: object | None,
) -> tuple[np.ndarray, int | None, list[object] | None]:
    batch_array = np.asarray(batch)
    fast_ref = _nonnegative_int_ref(ref_batch)
    if batch_array.dtype == np.dtype(np.int64) and not bool(np.any(batch_array < 0)):
        if ref_batch is None or fast_ref is not _NO_INT_REF:
            return batch_array, None if ref_batch is None else fast_ref, None
        raise ValueError(f"missing reference batch: {ref_batch!r}")

    labels = np.asarray(batch, dtype=object)
    if labels.ndim != 1:
        raise ValueError("batch must be a 1-dimensional vector")

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
