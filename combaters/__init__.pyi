from __future__ import annotations

from typing import Any, Protocol, TypedDict, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float64Matrix: TypeAlias = NDArray[np.float64]


class _DataFrameLike(Protocol):
    index: Any
    columns: Any


class _AnnDataLike(Protocol):
    X: Any


class _CombatReport(TypedDict):
    effective_mean_only: bool
    singleton_batches: list[Any]
    zero_variance_features: list[int]


class _ArrayCombatResult(TypedDict):
    adjusted: _Float64Matrix
    n_samples: int
    n_features: int
    report: _CombatReport


class _ObjectCombatResult(TypedDict):
    adjusted: Any
    n_samples: int
    n_features: int
    report: _CombatReport


@overload
def combat(
    values: _DataFrameLike,
    batch: Any,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: Any | None = None,
    *,
    formula: str | None = None,
) -> _ObjectCombatResult: ...


@overload
def combat(
    values: ArrayLike,
    batch: ArrayLike,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: Any | None = None,
    *,
    formula: str | None = None,
) -> _ArrayCombatResult: ...


def combat_frame(
    values: _DataFrameLike,
    batch: Any,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: Any | None = None,
    *,
    formula: str | None = None,
) -> _ObjectCombatResult: ...


def combat_anndata(
    adata: _AnnDataLike,
    batch: str | ArrayLike,
    *,
    layer: str | None = None,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: Any | None = None,
    formula: str | None = None,
) -> _ObjectCombatResult: ...
