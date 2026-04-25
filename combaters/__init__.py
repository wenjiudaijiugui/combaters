from __future__ import annotations

from typing import Any

import numpy as np

from ._combaters import combat as _combat


def combat(
    values: object,
    batch: object,
    mod: object | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]:
    """Run dense ComBat on a NumPy matrix or supported Python data object."""
    if _is_pandas_dataframe(values):
        return combat_frame(values, batch, mod, par_prior, mean_only, ref_batch)
    if _needs_python_adapter(values, batch, mod):
        return _combat(
            _coerce_matrix_if_needed(values, "values"),
            _coerce_batch_if_needed(batch),
            None if mod is None else _coerce_matrix_if_needed(mod, "mod"),
            par_prior,
            mean_only,
            ref_batch,
        )
    return _combat(values, batch, mod, par_prior, mean_only, ref_batch)


def combat_frame(
    values: Any,
    batch: Any,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]:
    """Run ComBat on a pandas DataFrame and return adjusted values as a DataFrame."""
    pd = _require_pandas()
    if not isinstance(values, pd.DataFrame):
        raise TypeError("combat_frame requires values to be a pandas DataFrame")

    _check_index_matches(values.index, batch, "batch")
    if _is_pandas_dataframe(mod):
        _check_index_matches(values.index, mod, "mod")

    result = _combat(
        _matrix_copy(values, "values"),
        _batch_copy(batch),
        None if mod is None else _matrix_copy(mod, "mod"),
        par_prior,
        mean_only,
        ref_batch,
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
    ref_batch: int | None = None,
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
    )
    if isinstance(result["adjusted"], np.ndarray):
        frame = _anndata_frame(result["adjusted"], adata)
        if frame is not None:
            result["adjusted"] = frame
    return result


def _needs_python_adapter(values: Any, batch: Any, mod: Any | None) -> bool:
    return (
        _is_scipy_sparse(values)
        or _is_scipy_sparse(mod)
        or _is_pandas_dataframe(mod)
        or _is_pandas_series(batch)
    )


def _coerce_matrix_if_needed(values: Any, name: str) -> Any:
    if _is_scipy_sparse(values) or _is_pandas_dataframe(values):
        return _matrix_copy(values, name)
    return values


def _coerce_batch_if_needed(batch: Any) -> Any:
    if _is_pandas_series(batch):
        return _batch_copy(batch)
    return batch


def _matrix_copy(values: Any, name: str) -> np.ndarray:
    if _is_scipy_sparse(values):
        dense = values.toarray()
        matrix = np.array(dense, dtype=np.float64, order="C", copy=True)
    else:
        matrix = np.array(values, dtype=np.float64, order="C", copy=True)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return matrix


def _batch_copy(batch: Any) -> np.ndarray:
    values = np.array(batch, dtype=np.int64, copy=True)
    if values.ndim != 1:
        raise ValueError("batch must be one-dimensional")
    return np.ascontiguousarray(values)


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
    if values is None:
        return False
    try:
        pd = _require_pandas()
    except ImportError:
        return False
    return isinstance(values, pd.DataFrame)


def _is_pandas_series(values: Any) -> bool:
    if values is None:
        return False
    try:
        pd = _require_pandas()
    except ImportError:
        return False
    return isinstance(values, pd.Series)


def _is_scipy_sparse(values: Any) -> bool:
    if values is None:
        return False
    try:
        from scipy import sparse
    except ImportError:
        return False
    return sparse.issparse(values)


def _require_pandas() -> Any:
    import pandas as pd

    return pd


__all__ = ["combat", "combat_anndata", "combat_frame"]
