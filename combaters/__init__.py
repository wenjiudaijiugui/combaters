from __future__ import annotations

from ._combaters import combat as _combat


def combat(
    values: object,
    batch: object,
    mod: object | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> dict[str, object]:
    """Run dense ComBat on a C-contiguous float64 NumPy matrix."""
    return _combat(values, batch, mod, par_prior, mean_only, ref_batch)


__all__ = ["combat"]
