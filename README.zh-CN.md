# combaters

[![Python](https://img.shields.io/badge/python-3.10--3.14-blue)](https://www.python.org/)
[![PyO3](https://img.shields.io/badge/PyO3-abi3--py310-orange)](https://pyo3.rs/)
[![Rust](https://img.shields.io/badge/core-Rust-dea584)](https://www.rust-lang.org/)
[![sva::ComBat](https://img.shields.io/badge/rewrite-sva%3A%3AComBat-4b8bbe)](https://bioconductor.org/packages/release/bioc/html/sva.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](https://opensource.org/licenses/MIT)

[English README](README.md)

`combaters` 是 Bioconductor `sva::ComBat` 的 Rust/PyO3 重写，面向 Python
中的密集矩阵 ComBat 批次效应校正。它保留熟悉的 ComBat 行为，同时把数值核心
放到 Rust 中，以获得更可预测的打包、内存占用和运行性能。

公开矩阵约定为行优先的 `samples x features`：
`values[sample * n_features + feature]`。

## Python 兼容性

发布 wheel 面向 CPython 3.10 到 3.14。扩展模块使用 PyO3 `abi3-py310`
构建；只有在锁定的 PyO3 版本支持更新 Python 小版本后，才应扩展该范围。

## Python API

```python
import numpy as np
from combaters import combat

values = np.asarray(..., dtype=np.float64).reshape((n_samples, n_features))
batch = np.asarray(...)
mod = np.asarray(..., dtype=np.float64).reshape((n_samples, n_covariates))

result = combat(values, batch, mod=mod, par_prior=True, mean_only=False, ref_batch=None)
adjusted = result["adjusted"]
```

所有公开矩阵输入都使用 `(n_samples, n_features)` 形状：行是样本，列是特征。
Python 包装层会把数值数组转换成 C 连续的 `float64` 后再进入 Rust 核心。

### 参数

| 参数 | 入口 | 说明 |
| --- | --- | --- |
| `values` | `combat`, `combat_frame` | 样本 x 特征数据。`combat` 接受列表、元组、NumPy 数组、pandas `DataFrame` 和 SciPy 稀疏矩阵；稀疏矩阵会被显式转成密集矩阵。`combat_frame` 要求 pandas `DataFrame`。 |
| `adata` | `combat_anndata` | AnnData-like 对象，从 `adata.X` 或 `adata.layers[layer]` 读取；不会原地修改对象。 |
| `batch` | 全部 | 长度为 `n_samples` 的一维批次标签。支持字符串、负整数、类别和 strided 数组。`combat_anndata` 也支持传入 `obs` 列名。 |
| `mod` | 全部 | 可选样本协变量，行数必须为 `n_samples`。数值列直接使用；DataFrame-like 的非数值列会做 dummy coding，并丢弃第一个观测水平作为参考。 |
| `formula` | 全部 | 可选 patsy 公式，用于构造 `mod`；需要可选的 `patsy` 支持和 `mod` 数据。 |
| `par_prior` | 全部 | `True` 使用参数化 empirical Bayes；`False` 使用非参数 empirical Bayes。 |
| `mean_only` | 全部 | `True` 只校正批次位置，不做尺度校正。单样本批次和退化特征也可能触发实际 mean-only 行为。 |
| `ref_batch` | 全部 | 可选参考批次，使用原始批次标签。参考批次对应的行会保持不变。 |
| `layer` | `combat_anndata` | 可选 AnnData layer 名称。`None` 读取 `adata.X`。 |

### 返回值

所有入口都返回一个字典，包含 `adjusted`、`n_samples`、`n_features` 和
`report`。`combat` 对数组类输入返回 NumPy 数组；当 `values` 是 `DataFrame`
时保留 pandas 标签。`combat_frame` 总是以 `DataFrame` 返回 `adjusted`。
`combat_anndata` 在 pandas 和 AnnData 标签可用时返回 `DataFrame`，否则返回
NumPy 数组。

```python
from combaters import combat_frame

result = combat_frame(values_df, batch_series)
adjusted_df = result["adjusted"]
```

```python
from combaters import combat_anndata

result = combat_anndata(adata, "batch", layer=None)
adjusted = result["adjusted"]
```

安装 `combaters[ecosystem]` 可拉取可选的 pandas 和 SciPy 辅助能力。

当安装了 `patsy` 时，可以使用 `formula`：

```python
result = combat(values, batch, mod=metadata, formula="~ age + C(treatment)")
```

`values` 中的缺失值会在拟合时忽略，并在 `adjusted` 中保留；无限值会被拒绝。
如果某个特征在任意多样本批次内零方差，该特征会原样复制回结果，并在
`result["report"]["zero_variance_features"]` 中报告。`prior.plots` 和
`BPPARAM` 不暴露；绘图不实现，并行执行由 Rust 核心自动决定。

## 并行执行

并行在 Rust 核心中自动进行，不提供 Python 或 R 风格的 `BPPARAM` API。
小矩阵走串行路径。较大矩阵在至少 65,536 个 cell 且至少 64 个独立
feature-by-batch job 时使用 Rayon。

并行循环对特征选择、投影、后验拟合、校正和特征回填都写入固定输出位置，
因此同一输入的结果是确定的。仅用于运行测试时，`COMBATERS_PARALLEL=off`
会强制串行路径，`COMBATERS_PARALLEL=parallel` 会强制并行路径；未设置或
设为 `auto` 时使用基于数据规模的自动策略。

## Rust 布局

- `crates/combaters-core`：纯 Rust ComBat 核心
- `src/lib.rs`：轻量 PyO3 绑定层
- `combaters/`：Python 包装层

## 引用

如果使用 `combaters`，请引用原始 ComBat 方法，以及提供参考实现
`sva::ComBat` 的 Bioconductor `sva` 包：

```bibtex
@article{johnson2007combat,
  title = {Adjusting batch effects in microarray expression data using empirical Bayes methods},
  author = {Johnson, W. Evan and Li, Cheng and Rabinovic, Ariel},
  journal = {Biostatistics},
  volume = {8},
  number = {1},
  pages = {118--127},
  year = {2007},
  doi = {10.1093/biostatistics/kxj037}
}

@article{leek2012sva,
  title = {The sva package for removing batch effects and other unwanted variation in high-throughput experiments},
  author = {Leek, Jeffrey T. and Johnson, W. Evan and Parker, Hilary S. and Jaffe, Andrew E. and Storey, John D.},
  journal = {Bioinformatics},
  volume = {28},
  number = {6},
  pages = {882--883},
  year = {2012},
  doi = {10.1093/bioinformatics/bts034}
}
```

- ComBat 方法：<https://doi.org/10.1093/biostatistics/kxj037>
- `sva` 包：<https://doi.org/10.1093/bioinformatics/bts034>
- Bioconductor `sva`：<https://bioconductor.org/packages/release/bioc/html/sva.html>
