from __future__ import annotations

from numpy.typing import ArrayLike


def combat(
    values: ArrayLike,
    batch: ArrayLike,
    mod: ArrayLike | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]: ...
