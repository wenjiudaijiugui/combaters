from __future__ import annotations

import numpy as np
import pytest


def balanced_matrix() -> np.ndarray:
    values = [
        4.0,
        1.0,
        7.0,
        2.0,
        5.0,
        1.5,
        8.0,
        2.5,
        6.5,
        2.0,
        8.8,
        3.0,
        11.0,
        7.0,
        2.0,
        6.0,
        12.5,
        7.5,
        2.5,
        6.5,
        13.0,
        8.0,
        3.0,
        7.0,
    ]
    return np.asarray(values, dtype=np.float64).reshape((6, 4))


def balanced_batch() -> np.ndarray:
    return np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64)


def test_combat_frame_preserves_dataframe_labels() -> None:
    pd = pytest.importorskip("pandas")
    from combaters import combat, combat_frame

    index = pd.Index([f"sample_{idx}" for idx in range(6)], name="sample")
    columns = pd.Index([f"feature_{idx}" for idx in range(4)], name="feature")
    values = pd.DataFrame(balanced_matrix(), index=index, columns=columns)
    batch = pd.Series(balanced_batch(), index=index, name="batch")
    mod = pd.DataFrame({"covariate": np.linspace(0.0, 1.0, 6)}, index=index)

    expected = combat(
        balanced_matrix(),
        balanced_batch(),
        mod=mod.to_numpy(dtype=np.float64),
    )["adjusted"]
    result = combat_frame(values, batch, mod=mod)
    auto_result = combat(values, batch, mod=mod)

    for adjusted in (result["adjusted"], auto_result["adjusted"]):
        assert isinstance(adjusted, pd.DataFrame)
        pd.testing.assert_index_equal(adjusted.index, index)
        pd.testing.assert_index_equal(adjusted.columns, columns)
        np.testing.assert_allclose(adjusted.to_numpy(), expected)


def test_combat_frame_rejects_misaligned_batch_index() -> None:
    pd = pytest.importorskip("pandas")
    from combaters import combat_frame

    index = pd.Index([f"sample_{idx}" for idx in range(6)])
    values = pd.DataFrame(balanced_matrix(), index=index)
    batch = pd.Series(balanced_batch(), index=list(reversed(index)))

    with pytest.raises(ValueError, match="batch index"):
        combat_frame(values, batch)


def test_combat_sparse_input_matches_dense_path() -> None:
    sparse = pytest.importorskip("scipy.sparse")
    from combaters import combat

    matrix = balanced_matrix()
    batch = balanced_batch()
    sparse_matrix = sparse.csr_matrix(matrix)

    result = combat(sparse_matrix, batch)
    expected = combat(matrix, batch)["adjusted"]

    assert isinstance(result["adjusted"], np.ndarray)
    np.testing.assert_allclose(result["adjusted"], expected)
    np.testing.assert_array_equal(sparse_matrix.toarray(), matrix)


def test_combat_anndata_reads_layer_and_obs_batch() -> None:
    pd = pytest.importorskip("pandas")
    from combaters import combat, combat_anndata

    class FakeAnnData:
        def __init__(self) -> None:
            self.obs_names = pd.Index([f"cell_{idx}" for idx in range(6)], name="cell")
            self.var_names = pd.Index([f"gene_{idx}" for idx in range(4)], name="gene")
            self.X = np.zeros((6, 4), dtype=np.float64)
            self.layers = {"counts": balanced_matrix()}
            self.obs = pd.DataFrame(
                {"batch": balanced_batch()},
                index=self.obs_names,
            )

    adata = FakeAnnData()

    result = combat_anndata(adata, "batch", layer="counts")
    expected = combat(balanced_matrix(), balanced_batch())["adjusted"]

    assert isinstance(result["adjusted"], pd.DataFrame)
    pd.testing.assert_index_equal(result["adjusted"].index, adata.obs_names)
    pd.testing.assert_index_equal(result["adjusted"].columns, adata.var_names)
    np.testing.assert_allclose(result["adjusted"].to_numpy(), expected)
    np.testing.assert_array_equal(adata.layers["counts"], balanced_matrix())
