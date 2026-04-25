from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike


def combat(
    values: ArrayLike,
    batch: ArrayLike,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: Any | None = None,
    *,
    formula: str | None = None,
) -> dict[str, object]: ...
