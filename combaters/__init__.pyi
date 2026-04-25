from __future__ import annotations

from typing import Any


def combat(
    values: Any,
    batch: Any,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]: ...


def combat_frame(
    values: Any,
    batch: Any,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]: ...


def combat_anndata(
    adata: Any,
    batch: str | Any,
    *,
    layer: str | None = None,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]: ...
