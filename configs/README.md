# 配置说明

`train.py` 的配置优先级：**默认值 < JSON 配置 < 命令行参数**。
JSON 里的键和命令行参数同名（下划线写法），例如
`train_res`、`texture_res`、`random_textures`、`normal_weight`。

## example.json

主线推荐配置，不含任何模型路径，配合 `--ref-mesh` 使用：

```bash
python train.py --config configs/example.json \
    --ref-mesh experiment/data/models/0001/mesh.obj \
    -o out/0001
```

流程会自动 QEM 简化 → 内置 UV 展开 → 优化。传入 `--base-mesh`
（以及可选的 `--base-mtl`）可以跳过前两步。

常用字段：

| 字段 | 含义 |
| --- | --- |
| `iter` | 优化迭代次数 |
| `batch` | 每次迭代随机视角数量 |
| `train_res` / `texture_res` | 训练分辨率 / 纹理分辨率 |
| `learning_rate` | Adam 学习率 |
| `loss` | `logl1`、`logl2`、`mse`、`smape`、`relativel2` |
| `random_textures` | 是否随机初始化纹理（没有参考贴图时设为 `true`） |
| `skip_train` | 不参与优化的项，如 `["kd", "ks"]` |
| `normal_weight` | 面法线损失权重（默认 `0.01`） |
| `laplacian_factor` | 拉普拉斯正则权重，`null` 表示自动估计 |
| `out_dir` | 输出目录，默认 `out/<模型目录名>` |

## upstream/

上游 nvdiffmodeling 的论文/示例配置，保留用于复现。它们都带有
`base_mesh`，因此会直接使用给定的初始网格，不触发 QEM 和 UV 展开。

这些配置引用 `data/` 下的官方 assets（README 中的 assets.zip，约 205 MB），
需要先下载并解压到项目根目录的 `data/`：

- `spot.json`：球体拟合 Spot（入门示例）
- `skull.json` / `skull_sphere.json` / `skull_reduce_paper.json`：骷髅法线与形状优化
- `ewer.json` / `ewer_paper.json`：从简化网格恢复花瓶
- `gardenia.json` / `gardenia_paper.json`、`hibiscus.json`：聚合几何
- `dancer_brushed_gold_64.json`：LOD，超采样参考
- `dancer_displacement.json` / `dancer_displacement_paper.json`：位移贴图

用法：

```bash
python train.py --config configs/upstream/spot.json
```

（`spot.json` 自带 `out_dir: "spot"`，输出到 `./spot`。）
