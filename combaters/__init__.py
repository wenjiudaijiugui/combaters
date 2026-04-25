from __future__ import annotations

import operator
from typing import Any

import numpy as np
import numpy.typing as npt

from ._combaters import combat as _combat

_NO_INT_REF = object()


def combat(
    values: npt.ArrayLike,
    batch: npt.ArrayLike,
    mod: object | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: object | None = None,
    *,
    formula: str | None = None,
) -> dict[str, object]:
    """Run dense ComBat after preparing Python-friendly inputs."""
    if _is_pandas_dataframe(values):
        return combat_frame(
            values,
            batch,
            mod=mod,
            par_prior=par_prior,
            mean_only=mean_only,
            ref_batch=ref_batch,
            formula=formula,
        )

    values_array = _float64_matrix("values", values)
    batch_array, ref_arg, levels = _prepare_batch(batch, ref_batch)
    mod_array = _prepare_mod(mod, formula)
    result = _combat(values_array, batch_array, mod_array, par_prior, mean_only, ref_arg)
    if levels is not None:
        _restore_report_labels(result, levels)
    return result


def combat_frame(
    values: Any,
    batch: Any,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: object | None = None,
    *,
    formula: str | None = None,
) -> dict[str, object]:
    """Run ComBat on a pandas DataFrame and return adjusted values as a DataFrame."""
    pd = _require_pandas()
    if not isinstance(values, pd.DataFrame):
        raise TypeError("combat_frame requires values to be a pandas DataFrame")

    _check_index_matches(values.index, batch, "batch")
    if _is_pandas_dataframe(mod) or _is_pandas_series(mod):
        _check_index_matches(values.index, mod, "mod")

    result = combat(
        _float64_matrix("values", values),
        batch,
        mod=mod,
        par_prior=par_prior,
        mean_only=mean_only,
        ref_batch=ref_batch,
        formula=formula,
    )
    result["adjusted"] = pd.DataFrame(
        result["adjusted"],
        index=values.index,
        columns=values.columns,
    )
    return result


def combat_anndata(
    adata: Any,
    batch: str | Any,
    *,
    layer: str | None = None,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: object | None = None,
    formula: str | None = None,
) -> dict[str, object]:
    """Run ComBat on a duck-typed AnnData object without mutating it."""
    values = _anndata_values(adata, layer)
    batch_values = _anndata_batch(adata, batch)
    result = combat(
        values,
        batch_values,
        mod=mod,
        par_prior=par_prior,
        mean_only=mean_only,
        ref_batch=ref_batch,
        formula=formula,
    )
    if isinstance(result["adjusted"], np.ndarray):
        frame = _anndata_frame(result["adjusted"], adata)
        if frame is not None:
            result["adjusted"] = frame
    return result


def _float64_matrix(name: str, value: npt.ArrayLike) -> np.ndarray:
    raw = value.toarray() if _is_scipy_sparse(value) else value
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 2D float64 array-like") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D float64 array-like")
    return np.ascontiguousarray(array, dtype=np.float64)


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


def _prepare_mod(mod: object | None, formula: str | None) -> np.ndarray | None:
    if formula is not None:
        return _prepare_formula_mod(mod, formula)
    if mod is None:
        return None
    if isinstance(mod, np.ndarray):
        return _float64_matrix("mod", mod)
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
    if _is_scipy_sparse(value):
        raw = value.toarray()  # type: ignore[attr-defined]
    elif hasattr(value, "to_numpy"):
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
        if not any(_labels_equal(value, category) for category in categories):
            categories.append(value)

    return [
        np.asarray([1.0 if _labels_equal(value, category) else 0.0 for value in values])
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


def _check_index_matches(index: Any, values: Any, name: str) -> None:
    other_index = getattr(values, "index", None)
    if other_index is not None and hasattr(other_index, "equals"):
        if len(other_index) == len(index) and not other_index.equals(index):
            raise ValueError(f"{name} index must match values index")


def _anndata_values(adata: Any, layer: str | None) -> Any:
    if layer is None:
        try:
            return adata.X
        except AttributeError as err:
            raise TypeError("combat_anndata requires an AnnData-like object with X") from err
    try:
        return adata.layers[layer]
    except AttributeError as err:
        raise TypeError("combat_anndata layer input requires adata.layers") from err
    except KeyError as err:
        raise KeyError(f"AnnData layer {layer!r} not found") from err


def _anndata_batch(adata: Any, batch: str | Any) -> Any:
    if not isinstance(batch, str):
        return batch
    try:
        return adata.obs[batch]
    except AttributeError as err:
        raise TypeError("string batch input requires adata.obs") from err
    except KeyError as err:
        raise KeyError(f"AnnData obs column {batch!r} not found") from err


def _anndata_frame(adjusted: np.ndarray, adata: Any) -> Any | None:
    try:
        pd = _require_pandas()
    except ImportError:
        return None
    index = getattr(adata, "obs_names", None)
    columns = getattr(adata, "var_names", None)
    if index is None or columns is None:
        return None
    return pd.DataFrame(adjusted, index=index, columns=columns)


def _is_pandas_dataframe(values: Any) -> bool:
    return _is_type_from_module(values, "pandas.", "DataFrame")


def _is_pandas_series(values: Any) -> bool:
    return _is_type_from_module(values, "pandas.", "Series")


def _is_scipy_sparse(values: Any) -> bool:
    return (
        values is not None
        and type(values).__module__.startswith("scipy.sparse")
        and hasattr(values, "toarray")
    )


def _is_type_from_module(values: Any, module_prefix: str, type_name: str) -> bool:
    module = type(values).__module__
    return (
        values is not None
        and type(values).__name__ == type_name
        and (module == module_prefix.rstrip(".") or module.startswith(module_prefix))
    )


def _require_pandas() -> Any:
    import pandas as pd

    return pd


__all__ = ["combat", "combat_anndata", "combat_frame"]
