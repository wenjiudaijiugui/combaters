from __future__ import annotations

import numpy as np

from ._combaters import combat as _combat


def combat(
    values: object,
    batch: object,
    mod: object | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
    *,
    formula: str | None = None,
) -> dict[str, object]:
    """Run dense ComBat on a C-contiguous float64 NumPy matrix."""
    prepared_mod = _prepare_mod(mod, formula)
    return _combat(values, batch, prepared_mod, par_prior, mean_only, ref_batch)


def _prepare_mod(mod: object | None, formula: str | None) -> object | None:
    if formula is not None:
        return _prepare_formula_mod(mod, formula)
    if mod is None or isinstance(mod, np.ndarray):
        return mod
    return _prepare_dataframe_like_mod(mod)


def _prepare_formula_mod(mod: object | None, formula: str) -> np.ndarray:
    if mod is None:
        raise ValueError("formula requires mod data")
    try:
        import patsy
    except ImportError as exc:
        raise ImportError("formula support requires the optional patsy package") from exc

    design = patsy.dmatrix(formula, mod, return_type="dataframe")
    return np.ascontiguousarray(np.asarray(design, dtype=np.float64))


def _prepare_dataframe_like_mod(mod: object) -> np.ndarray:
    raw = _to_numpy(mod)
    if raw.ndim == 1:
        columns = [raw]
    elif raw.ndim == 2:
        columns = [raw[:, column] for column in range(raw.shape[1])]
    else:
        raise ValueError("mod must be a 1D or 2D array-like object")

    n_samples = raw.shape[0]
    encoded_columns: list[np.ndarray] = []
    for column in columns:
        encoded_columns.extend(_encode_column(column))

    if not encoded_columns:
        return np.empty((n_samples, 0), dtype=np.float64)
    return np.ascontiguousarray(np.column_stack(encoded_columns), dtype=np.float64)


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        raw = value.to_numpy()  # type: ignore[attr-defined]
    else:
        raw = np.asarray(value)
    return np.asarray(raw)


def _encode_column(column: np.ndarray) -> list[np.ndarray]:
    numeric = _try_numeric_column(column)
    if numeric is not None:
        return [numeric]
    return _dummy_columns(column)


def _try_numeric_column(column: np.ndarray) -> np.ndarray | None:
    try:
        return np.asarray(column, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _dummy_columns(column: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(column, dtype=object)
    categories: list[object] = []
    for value in values:
        if _is_missing(value):
            raise ValueError("categorical mod columns must not contain missing values")
        if not any(_same_value(value, category) for category in categories):
            categories.append(value)

    return [
        np.asarray([1.0 if _same_value(value, category) else 0.0 for value in values])
        for category in categories[1:]
    ]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        return bool(np.isnan(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _same_value(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


__all__ = ["combat"]
