from __future__ import annotations

from typing import Any


def combat(
    values: Any,
    batch: Any,
    mod: Any | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
    *,
    formula: str | None = None,
) -> dict[str, object]: ...
