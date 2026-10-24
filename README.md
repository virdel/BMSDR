# BMS-DR

**BMS-DR**（**B**uilding **M**esh **S**implification with **D**ifferentiable **R**endering）
是一个面向建筑模型的可微渲染简化工程：用可微渲染拟合纹理与几何，把建筑高模简化为
带贴图的低模。它基于 NVIDIA [nvdiffmodeling](https://github.com/NVlabs/nvdiffmodeling)
（论文 *Appearance-Driven Automatic 3D Model Simplification*），在其之上做了精简：
去掉 Blender / pymeshlab / xatlas 依赖，内置纯 Python 的分块 UV 展开器，并把
实验性代码集中到 `dev` 分支。

适用模型：建筑、古建、城市部件等以硬表面/平面为主的网格（QEM 简化 + UV 展开 +
纹理/几何联合优化）。

## 分支说明

| 分支 | 内容 |
| --- | --- |
| `main` | 精简主线：QEM 简化 + 内置 UV 展开 + 训练流程，只保留运行所需代码 |
| `dev` | 历史/实验代码：Blender 与 xatlas 的 UV 方案、alpha shape、批量与联合训练脚本等 |

旧流程的代码没有删除，只是从 `main` 移到了 `dev`：

```bash
git checkout dev    # 查看旧的 Blender / alpha shape / 批量实验代码
git checkout main   # 回到精简主线
```

## 主要改动

1. **移除 Blender 依赖。** 原 `python_blender.py` 只是为了调用
   `bpy.ops.uv.smart_project`。现在由 [`src/uv_unwrap.py`](src/uv_unwrap.py)
   用标准库实现：
   - 按法线夹角（默认 66°）在网格上生长 UV 岛；
   - 把每个岛投影到平均平面；
   - 用 shelf packing 把岛排进 `[0,1]²`，岛与岛之间留 margin；
   - 输出 `mesh_alpha_uv.obj` / `.mtl`，顶点按岛复制，UV 连续且不跨岛重叠。
   对同一份 4.2 万面的 QEM 结果，Blender 生成 31780 个 UV 岛，本实现生成
   29478 个，效果相当，但不依赖外部程序、结果可复现。
2. **训练入口重构。** `train.py` 不再硬编码 `E:\...` 路径和批量实验开关，
   改为 `--ref-mesh / --base-mesh / --out-dir / --config` 参数；配置优先级为
   *默认值 < JSON 配置 < 命令行参数*。
3. **删除死代码。** 去掉未使用的 depth/chamfer loss、重复的 config 写入、
   大量注释掉的备选实现和调试输出。
4. **QEM 调用更安全。** [`src/simplify.py`](src/simplify.py) 使用
   `subprocess` 参数列表调用 `bin/qem/qem.exe`，路径带空格也不会出错。
5. **OBJ 预处理修复。** [`src/check_obj_file.py`](src/check_obj_file.py)
   流式读写，保持原文件编码（UTF-8/GBK），不再把 GBK 模型强制改写成 UTF-8。

## 环境

- Windows（QEM 可执行文件为 Windows 版本）
- Python 3.8+（原始 README 的 3.6 也可运行）
- CUDA + 支持 CUDA 的 PyTorch
- [nvdiffrast](https://nvlabs.github.io/nvdiffrast/)（按官方说明从源码安装）
- 其余依赖见 [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
# 再按 https://nvlabs.github.io/nvdiffrast/ 安装 nvdiffrast
```

## 快速开始

### 完整流程：参考模型 -> QEM 简化 -> UV 展开 -> 纹理/几何优化

```bash
python train.py --config configs/example.json --ref-mesh data/buildings/0001/mesh.obj -o out/0001
```

不传 `-o` 时输出到 `out/<模型目录名>/`。流程会自动：

1. 清理参考 OBJ 中 CGAL 不支持的 `l` 记录；
2. 调用 `bin/qem/qem.exe` 生成 `mesh_qem_3000.obj`；
3. 调用内置 UV 展开生成 `mesh_alpha_uv.obj` / `.mtl`；
4. 运行可微渲染优化，输出 `mesh/`、`mesh_initial/`、损失曲线和 `config.json`。

### 已有初始网格（跳过简化和 UV 展开）

```bash
python train.py --ref-mesh data/spot/spot.obj --base-mesh data/sphere.obj --base-mtl data/sphere.mtl -o out/spot
```

`--base-mesh` 必须自带 UV（或通过 `--base-mtl` 指定材质）。

### 只做 UV 展开

```bash
python -m src.uv_unwrap input.obj out_dir --name mesh_alpha_uv --angle 66 --margin 0.002
```

参数：

- `--angle`：同一 UV 岛内相邻面法线最大夹角（度），越小岛越多、纹理越碎；
- `--margin`：岛间留白（UV 单位）。

### 只做 QEM 简化

```bash
python -m src.simplify input.obj out_dir
```

### 测试

UV 展开器只依赖标准库，可以直接跑单元测试：

```bash
python -m unittest discover -s tests -v
```

## 配置与参数

命令行参数与 JSON 配置同名（下划线/连字符均可），例如
`--train-res`、`--texture-res 1024 1024`、`--normal-weight 0.01`。
配置里可以写 `ref_mesh`、`base_mesh`、`out_dir`，命令行会覆盖配置。

## 数据说明

把待处理模型放到 `data/` 下（例如 `data/buildings/0001/mesh.obj`）即可；
模型、训练输出与实验数据（如 `data/`、`out/`，合计可达数十 GB）都不纳入 Git
跟踪，已在 [`.gitignore`](.gitignore) 中整体忽略；文件仍保留在本地磁盘。
`main` 只跟踪代码、配置和文档。

## 目录

```
train.py              训练/优化入口
src/uv_unwrap.py      内置 UV 展开（无第三方依赖）
src/simplify.py       QEM 简化封装
src/check_obj_file.py OBJ 预处理与统计
src/                  可微渲染核心（上游代码）
bin/qem/              QEM 可执行文件
configs/              示例配置（example.json + upstream/ 原始论文配置，见 configs/README.md）
tests/                UV 展开单元测试
dev 分支              Blender/xatlas UV、alpha shape、批量/联合训练等历史代码
```

## License & Citation

Copyright © 2021, NVIDIA Corporation. 本工程基于 NVIDIA Source Code License 发布，
详见 [LICENSE.txt](LICENSE.txt)。

```
@inproceedings{Hasselgren2021,
  title     = {Appearance-Driven Automatic 3D Model Simplification},
  author    = {Jon Hasselgren and Jacob Munkberg and Jaakko Lehtinen and Miika Aittala and Samuli Laine},
  booktitle = {Eurographics Symposium on Rendering},
  year      = {2021}
}
```
