# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""BMS-DR 训练入口：用可微渲染把建筑低模拟合到高模。

整体流程
--------
1. ``main`` 读取配置，清理参考 OBJ，准备初始低模：
   - 默认调用 QEM 简化 + 内置 UV 展开；
   - 也可通过 ``--base-mesh`` 直接提供带 UV 的低模。
2. ``optimize_mesh`` 用 nvdiffrast 渲染参考高模和待优化低模，
   把渲染图像差、拉普拉斯平滑项、面法线一致项相加作为损失。
3. 反向传播更新顶点位置与 kd/ks/normal 贴图，迭代结束后导出：
   - ``mesh/``：优化后的低模与贴图；
   - ``mesh_initial/``：优化前的低模，便于对比；
   - ``img_*.png``、损失曲线和 ``loss.csv``。

约定
----
* 训练全部在 CUDA 上完成（nvdiffrast 需要 OpenGL/CUDA 上下文）。
* ``FLAGS`` 是 argparse.Namespace，字段来自“默认值 + JSON 配置 + 命令行”，
  沿用上游代码的命名。
* 参考模型（ref_mesh）保持原始分辨率；低模（base_mesh）经过 QEM/UV
  处理后顶点顺序与参考模型无关，只共享同一世界坐标系。
"""

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

# 同时加载 torch 和 MKL 时避免 OpenMP 重复初始化导致进程退出。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import nvdiffrast.torch as dr

import src.renderutils as ru
from src import obj
from src import util
from src import mesh
from src import texture
from src import render
from src import regularizer
from src.check_obj_file import check_obj_file
from src.mesh import Mesh
from src.simplify import qem_simplify
from src.uv_unwrap import generate_uv

# 相机到原点的距离；随机视角采样和预览相机都基于它。
RADIUS = 3.5
# 渲染前把模型居中并缩放到该尺度，避免远近裁剪面随模型尺寸失效。
MESH_SCALE = 2.0
# 学习率衰减系数：lr = 10^(-step * 0.0002)，5000 步后约剩 10%。
LR_DECAY = 0.0002
# 拉普拉斯权重的自动估计：LAP_AUTO_SCALE * (0.1 / lap_loss)。
LAP_AUTO_SCALE = 0.25
# 拉普拉斯权重的下限比例，避免后期完全失去平滑约束。
LAP_MIN_RATIO = 0.15


###############################################################################
# 基础工具
###############################################################################

def load_mesh(filename, mtl_override=None):
    """加载 OBJ 网格（顶点、UV、法线和材质一起读入）。"""
    _, ext = os.path.splitext(filename)
    if ext.lower() != ".obj":
        raise ValueError("Only .obj meshes are supported, got: %s" % filename)
    return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)


def create_loss(FLAGS):
    """根据配置里的名字构造图像损失函数。

    使用对数色调映射的 l1/l2 对暗部更敏感，适合建筑纹理里的大面积
    阴影和室内区域；smape/relativel2 为可选的相对误差。
    """
    presets = {
        "smape": ("smape", "none"),
        "mse": ("mse", "none"),
        "logl1": ("l1", "log_srgb"),
        "logl2": ("mse", "log_srgb"),
        "relativel2": ("relmse", "none"),
    }
    if FLAGS.loss not in presets:
        raise ValueError("Unknown loss: %s" % FLAGS.loss)
    loss_name, tonemapper = presets[FLAGS.loss]

    def loss_fn(img, ref):
        return ru.image_loss(img, ref, loss=loss_name, tonemapper=tonemapper)

    return loss_fn


def _check_texture_resolution(FLAGS, ref_mesh):
    """优化贴图比参考贴图还小时给出警告（会丢失高频细节）。"""
    ref_res = np.maximum(
        ref_mesh.material["kd"].getRes(), ref_mesh.material["ks"].getRes()
    )
    if "normal" in ref_mesh.material:
        ref_res = np.maximum(ref_res, ref_mesh.material["normal"].getRes())
    if FLAGS.texture_res[0] < ref_res[0] or FLAGS.texture_res[1] < ref_res[1]:
        print(
            "---> WARNING: Picked a texture resolution lower than the reference "
            "mesh [%d, %d] < [%d, %d]"
            % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_res[0], ref_res[1])
        )


def _create_trainable_textures(FLAGS, ref_mesh):
    """创建三张可训练贴图：法线、基础色 kd、高光/粗糙度 ks。

    ``random_textures=True`` 时随机初始化，否则从参考模型的贴图开始；
    参考模型没有法线贴图时，法线贴图从 (0, 0, 1) 开始。
    """
    trainable_mips = not FLAGS.custom_mip

    if FLAGS.random_textures or "normal" not in ref_mesh.material:
        normal_map = texture.create_trainable(
            np.array([0, 0, 1]), FLAGS.texture_res, trainable_mips
        )
    else:
        normal_map = texture.create_trainable(
            ref_mesh.material["normal"], FLAGS.texture_res, trainable_mips
        )

    if FLAGS.random_textures:
        channels = 4 if FLAGS.layers > 1 else 3
        kd_map = texture.create_trainable(
            np.random.uniform(
                size=FLAGS.texture_res + [channels], low=0.0, high=1.0
            ),
            FLAGS.texture_res,
            trainable_mips,
        )
        ks_r = np.random.uniform(
            size=FLAGS.texture_res + [1], low=0.0, high=0.01
        )
        ks_g = np.random.uniform(
            size=FLAGS.texture_res + [1], low=FLAGS.min_roughness, high=1.0
        )
        ks_b = np.random.uniform(
            size=FLAGS.texture_res + [1], low=0.0, high=1.0
        )
        ks_map = texture.create_trainable(
            np.concatenate((ks_r, ks_g, ks_b), axis=2),
            FLAGS.texture_res,
            trainable_mips,
        )
    else:
        kd_map = texture.create_trainable(
            ref_mesh.material["kd"], FLAGS.texture_res, trainable_mips
        )
        ks_map = texture.create_trainable(
            ref_mesh.material["ks"], FLAGS.texture_res, trainable_mips
        )

    return normal_map, kd_map, ks_map


def _build_trainable_meshes(
    FLAGS,
    v_pos_opt,
    normalized_base_mesh,
    ref_mesh,
    opt_material,
    displacement_map_var,
):
    """构建可训练低模；需要位移贴图时再做细分和位移。"""
    # 用可训练的顶点坐标替换归一化低模的位置，拓扑/UV 沿用低模。
    opt_base_mesh = Mesh(
        v_pos_opt,
        normalized_base_mesh.t_pos_idx,
        material=opt_material,
        base=normalized_base_mesh,
    )

    # 从 [-1, 1] 的归一化空间对齐到参考模型的实际包围盒。
    opt_base_mesh = mesh.align_with_reference(opt_base_mesh, ref_mesh)
    # 重新计算平滑顶点法线，并建立切线空间供法线贴图采样。
    opt_base_mesh = mesh.auto_normals(opt_base_mesh)
    opt_base_mesh = mesh.compute_tangents(opt_base_mesh)

    if FLAGS.subdivision > 0:
        subdivided = mesh.subdivide(opt_base_mesh, steps=FLAGS.subdivision)
        opt_detail_mesh = mesh.displace(
            subdivided,
            displacement_map_var,
            FLAGS.displacement,
            keep_connectivity=True,
        )
    else:
        opt_detail_mesh = opt_base_mesh

    return opt_base_mesh, opt_detail_mesh


def _create_regularizers(FLAGS, opt_base_mesh, opt_detail_mesh):
    """构造几何正则项。

    - 拉普拉斯项：抑制顶点乱飞，保持三角形形状和表面平滑；
    - 面法线项：抑制相邻三角形的法线突变，避免简化后出现折痕/翻面。
    """
    if FLAGS.relative_laplacian:
        with torch.no_grad():
            reference = opt_base_mesh.eval().clone()
        lap_loss_fn = regularizer.laplace_regularizer_const(
            opt_detail_mesh, reference
        )
    else:
        lap_loss_fn = regularizer.laplace_regularizer_const(opt_detail_mesh)

    normal_loss_fn = regularizer.face_normal_regularizer(opt_detail_mesh)
    return lap_loss_fn, normal_loss_fn


def _create_optimizer(FLAGS, trainable_list):
    """Adam + 指数学习率衰减。"""
    optimizer = torch.optim.Adam(trainable_list, lr=FLAGS.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(0.0, 10 ** (-step * LR_DECAY))
    )
    return optimizer, scheduler


def _create_background(FLAGS):
    """生成训练时使用的背景色（白/棋盘格/无）。"""
    if FLAGS.background == "checker":
        return torch.tensor(
            util.checkerboard(FLAGS.display_res, 8),
            dtype=torch.float32,
            device="cuda",
        )
    if FLAGS.background == "white":
        return torch.ones(
            (1, FLAGS.display_res, FLAGS.display_res, 3),
            dtype=torch.float32,
            device="cuda",
        )
    return None


def _training_resolution(FLAGS):
    """返回当前迭代的训练分辨率和 spp。"""
    resolution = FLAGS.train_res
    spp = FLAGS.spp
    if FLAGS.random_train_res:
        # 随机分辨率，但保持“分辨率 × spp”近似不变，稳定每步的采样量。
        resolution = np.random.randint(16, FLAGS.train_res + 1)
        spp = FLAGS.spp * (FLAGS.train_res // resolution)
    return resolution, spp


def _sample_camera_batch(FLAGS, proj_mtx):
    """为当前迭代采样 ``FLAGS.batch`` 组随机视角和光源位置。"""
    mvp = np.zeros((FLAGS.batch, 4, 4), dtype=np.float32)
    campos = np.zeros((FLAGS.batch, 3), dtype=np.float32)
    lightpos = np.zeros((FLAGS.batch, 3), dtype=np.float32)

    for batch_index in range(FLAGS.batch):
        rotation = util.random_rotation_translation(0.25)
        model_view = np.matmul(
            util.translate(0, 0, -RADIUS), rotation
        )
        mvp[batch_index] = np.matmul(proj_mtx, model_view).astype(np.float32)
        campos[batch_index] = np.linalg.inv(model_view)[:3, 3]
        # 光源围绕相机做余弦采样，保证高光和阴影有变化。
        lightpos[batch_index] = util.cosine_sample(campos[batch_index]) * RADIUS
    return mvp, campos, lightpos


def _maybe_save_preview(
    FLAGS,
    glctx,
    opt_base_mesh,
    opt_detail_mesh,
    render_ref_mesh,
    ref_mesh_aabb,
    mesh_scale,
    proj_mtx,
    background,
    out_dir,
    iteration,
    img_cnt,
    angle_deg,
):
    """按间隔渲染“优化结果 / 参考模型”对比图，返回更新后的计数和角度。

    相机绕 Y 轴每渲染一次前进 5°，因此连续保存的图片能覆盖模型四周。
    """
    display_image = bool(
        FLAGS.display_interval and iteration % FLAGS.display_interval == 0
    )
    save_image = bool(
        FLAGS.save_interval and iteration % FLAGS.save_interval == 0
    )
    if not (display_image or save_image):
        return img_cnt, angle_deg

    eye = np.array(
        [
            RADIUS * math.cos(math.radians(angle_deg)),
            0.0,
            RADIUS * math.sin(math.radians(angle_deg)),
        ]
    )
    angle_deg += 5
    up = np.array(FLAGS.camera_up)
    at = np.array([0, 0, 0])
    model_view = util.lookAt(eye, at, up)
    mvp = np.matmul(proj_mtx, model_view).astype(np.float32)[None, ...]

    # 预览也用“相机位置”当光源，和训练时的 headlight 一致。
    camera_position = np.linalg.inv(model_view)[None, :3, 3]
    params = {
        "mvp": mvp,
        "lightpos": camera_position,
        "campos": camera_position,
        "resolution": [FLAGS.display_res, FLAGS.display_res],
        "time": 0,
    }

    with torch.no_grad():
        detail_mesh = mesh.center_by_reference(
            opt_detail_mesh.eval(params), ref_mesh_aabb, mesh_scale
        )
        reference_mesh = mesh.center_by_reference(
            render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale
        )

        if FLAGS.subdivision > 0:
            base_mesh = mesh.center_by_reference(
                opt_base_mesh.eval(params), ref_mesh_aabb, mesh_scale
            )
            img_base = render.render_mesh(
                glctx,
                base_mesh,
                mvp,
                camera_position,
                camera_position,
                FLAGS.light_power,
                FLAGS.display_res,
                num_layers=FLAGS.layers,
                background=background,
                min_roughness=FLAGS.min_roughness,
            )[0]
            img_base = util.scale_img_nhwc(
                img_base, [FLAGS.display_res, FLAGS.display_res]
            )

        img_opt = render.render_mesh(
            glctx,
            detail_mesh,
            mvp,
            camera_position,
            camera_position,
            FLAGS.light_power,
            FLAGS.display_res,
            num_layers=FLAGS.layers,
            background=background,
            min_roughness=FLAGS.min_roughness,
        )[0]
        img_ref = render.render_mesh(
            glctx,
            reference_mesh,
            mvp,
            camera_position,
            camera_position,
            FLAGS.light_power,
            FLAGS.display_res,
            num_layers=1,
            spp=FLAGS.spp,
            background=background,
            min_roughness=FLAGS.min_roughness,
        )[0]

        img_opt = util.scale_img_nhwc(
            img_opt, [FLAGS.display_res, FLAGS.display_res]
        )
        img_ref = util.scale_img_nhwc(
            img_ref, [FLAGS.display_res, FLAGS.display_res]
        )

        if FLAGS.subdivision > 0:
            result_image = torch.cat([img_base, img_opt, img_ref], axis=2)
        else:
            result_image = torch.cat([img_opt, img_ref], axis=2)

    result_image[0] = util.tonemap_srgb(result_image[0])
    np_result_image = result_image[0].detach().cpu().numpy()

    if display_image:
        util.display_image(
            np_result_image,
            size=FLAGS.display_res,
            title="%d / %d" % (iteration, FLAGS.iter),
        )
    if save_image:
        util.save_image(
            os.path.join(out_dir, "img_%06d.png" % img_cnt), np_result_image
        )
        img_cnt += 1

    return img_cnt, angle_deg


def _save_training_curves(
    out_dir,
    img_loss_vec,
    total_loss_vec,
    lap_fac_vec,
    learning_rate_vec,
    normal_loss_vec,
):
    """把训练曲线保存为 jpg，便于快速检查收敛情况。"""
    plt.figure()
    plt.plot(img_loss_vec, label="img_loss")
    plt.plot(total_loss_vec, label="total_loss")
    plt.legend(loc=0)
    plt.savefig(os.path.join(out_dir, "total_loss.jpg"))
    plt.close()

    for filename, values, label in (
        ("lap_fac.jpg", lap_fac_vec, "lap_fac"),
        ("learning_rate.jpg", learning_rate_vec, "learning_rate"),
        ("normal_loss.jpg", normal_loss_vec, "normal_loss"),
    ):
        plt.figure()
        plt.plot(values, label=label)
        plt.legend(loc=0)
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()


def _save_loss_data(
    out_dir,
    img_loss_vec,
    total_loss_vec,
    lap_loss_vec,
    learning_rate_vec,
    normal_loss_vec,
    lap_fac_vec,
    iter_dur_vec,
):
    """保存逐迭代损失 CSV 和 pickle（后者供旧脚本读取）。"""
    dataframe = pd.DataFrame(
        {
            "img_loss": img_loss_vec,
            "total_loss": total_loss_vec,
            "learning_rate": learning_rate_vec,
            "normal_loss": normal_loss_vec,
            "lap_fac": lap_fac_vec,
            "time": iter_dur_vec,
        }
    )
    dataframe.to_csv(os.path.join(out_dir, "loss.csv"))

    with open(os.path.join(out_dir, "loss_vec.txt"), "wb") as handle:
        pickle.dump([img_loss_vec, lap_loss_vec], handle)


###############################################################################
# 优化主循环
###############################################################################

def optimize_mesh(FLAGS, out_dir, log_interval=10, mesh_scale=MESH_SCALE):
    """执行可微渲染优化。

    Args:
        FLAGS: 合并后的配置（Namespace）。
        out_dir: 输出目录，保存图片、网格、贴图和损失曲线。
        log_interval: 控制台日志间隔。
        mesh_scale: 居中后缩放到的渲染尺度。
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mesh"), exist_ok=True)

    # 投影矩阵：x=0.4 为水平视场角的正切，f=1000 为远裁剪面。
    proj_mtx = util.projection(x=0.4, f=1000.0)

    if FLAGS.learning_rate is None:
        FLAGS.learning_rate = 0.01

    assert not FLAGS.random_train_res or FLAGS.custom_mip, (
        "Random training resolution requires custom mip."
    )

    # ------------------------------------------------------------------
    # 1. 读取参考模型和初始低模
    # ------------------------------------------------------------------
    ref_mesh = load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
    print(
        "Ref mesh has %d triangles and %d vertices."
        % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0])
    )
    _check_texture_resolution(FLAGS, ref_mesh)

    base_mesh = load_mesh(FLAGS.base_mesh, FLAGS.base_mtl)
    print(
        "Base mesh has %d triangles and %d vertices."
        % (base_mesh.t_pos_idx.shape[0], base_mesh.v_pos.shape[0])
    )
    print("Avg edge length: %f" % regularizer.avg_edge_length(base_mesh))

    # 统一缩放到单位尺寸：学习率、正则权重就不用随模型实际尺寸调整。
    normalized_base_mesh = mesh.unit_size(base_mesh)

    # ------------------------------------------------------------------
    # 2. 创建可训练变量：顶点位置 + 三张贴图（+ 可选位移贴图）
    # ------------------------------------------------------------------
    v_pos_opt = normalized_base_mesh.v_pos.clone().detach().requires_grad_(True)
    normal_map_opt, kd_map_opt, ks_map_opt = _create_trainable_textures(
        FLAGS, ref_mesh
    )

    displacement_map_var = None
    if FLAGS.subdivision > 0:
        displacement_map_var = torch.zeros(
            FLAGS.texture_res + [1],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )

    # ``skip_train`` 可以冻结某一项，只优化其余部分（例如只优化几何）。
    trainable_list = []
    if "position" not in FLAGS.skip_train:
        trainable_list += [v_pos_opt]
        print("train position")
    if "normal" not in FLAGS.skip_train:
        trainable_list += normal_map_opt.getMips()
        print("train normal")
    if "kd" not in FLAGS.skip_train:
        trainable_list += kd_map_opt.getMips()
        print("train kd")
    if "ks" not in FLAGS.skip_train:
        trainable_list += ks_map_opt.getMips()
        print("train ks")
    if "displacement" not in FLAGS.skip_train and displacement_map_var is not None:
        trainable_list += [displacement_map_var]
        print("train displacement")

    opt_material = {
        "bsdf": ref_mesh.material["bsdf"],
        "kd": kd_map_opt,
        "ks": ks_map_opt,
        "normal": normal_map_opt,
    }

    # ------------------------------------------------------------------
    # 3. 构建计算图：参考模型切线空间、可训练低模、正则项和优化器
    # ------------------------------------------------------------------
    render_ref_mesh = mesh.compute_tangents(ref_mesh)
    ref_mesh_aabb = mesh.aabb(render_ref_mesh.eval())

    opt_base_mesh, opt_detail_mesh = _build_trainable_meshes(
        FLAGS,
        v_pos_opt,
        normalized_base_mesh,
        ref_mesh,
        opt_material,
        displacement_map_var,
    )
    lap_loss_fn, face_normal_loss_fn = _create_regularizers(
        FLAGS, opt_base_mesh, opt_detail_mesh
    )
    optimizer, scheduler = _create_optimizer(FLAGS, trainable_list)
    image_loss_fn = create_loss(FLAGS)
    background = _create_background(FLAGS)

    # 保存初始低模，训练结束后可以和 ``mesh/`` 直接对比。
    os.makedirs(os.path.join(out_dir, "mesh_initial"), exist_ok=True)
    obj.write_obj(os.path.join(out_dir, "mesh_initial/"), opt_base_mesh.eval())

    glctx = dr.RasterizeGLContext()

    # ------------------------------------------------------------------
    # 4. 训练循环
    # ------------------------------------------------------------------
    img_cnt = 0
    angle_deg = 0.0
    img_loss_vec = []
    lap_loss_vec = []
    total_loss_vec = []
    normal_loss_vec = []
    lap_fac_vec = []
    learning_rate_vec = []
    iter_dur_vec = []

    # 拉普拉斯权重在第 0 步估计，之后指数衰减；先占位避免未定义。
    lap_fac = 0.0
    min_lap_fac = 0.0

    for it in range(FLAGS.iter + 1):
        # 4.1 定期渲染对比图（it=0 时保存的是初始低模）。
        img_cnt, angle_deg = _maybe_save_preview(
            FLAGS,
            glctx,
            opt_base_mesh,
            opt_detail_mesh,
            render_ref_mesh,
            ref_mesh_aabb,
            mesh_scale,
            proj_mtx,
            background,
            out_dir,
            it,
            img_cnt,
            angle_deg,
        )

        iter_start_time = time.time()
        iter_res, iter_spp = _training_resolution(FLAGS)
        mvp, campos, lightpos = _sample_camera_batch(FLAGS, proj_mtx)
        params = {
            "mvp": mvp,
            "lightpos": lightpos,
            "campos": campos,
            "resolution": [iter_res, iter_res],
            "time": 0,
        }
        random_bg_color = torch.rand(
            FLAGS.batch, iter_res, iter_res, 3, dtype=torch.float32, device="cuda"
        )

        # 4.2 把两个模型对齐到同一参考包围盒，再分别渲染。
        _opt_ref = mesh.center_by_reference(
            render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale
        )
        _opt_detail = mesh.center_by_reference(
            opt_detail_mesh.eval(params), ref_mesh_aabb, mesh_scale
        )

        # 参考模型不需要梯度，只生成“目标图像”。
        with torch.no_grad():
            color_ref, _ = render.render_mesh(
                glctx,
                _opt_ref,
                mvp,
                campos,
                lightpos,
                FLAGS.light_power,
                iter_res,
                spp=iter_spp,
                num_layers=1,
                background=random_bg_color,
                min_roughness=FLAGS.min_roughness,
            )

        color_opt, _ = render.render_mesh(
            glctx,
            _opt_detail,
            mvp,
            campos,
            lightpos,
            FLAGS.light_power,
            iter_res,
            spp=iter_spp,
            num_layers=FLAGS.layers,
            msaa=True,
            background=random_bg_color,
            min_roughness=FLAGS.min_roughness,
        )

        # 4.3 三项损失：图像外观 + 拉普拉斯平滑 + 面法线一致。
        img_loss = image_loss_fn(color_opt, color_ref)
        lap_loss = lap_loss_fn.eval(params)
        normal_loss = face_normal_loss_fn.eval(params)

        if it == 0:
            if FLAGS.laplacian_factor is not None:
                lap_fac = FLAGS.laplacian_factor
            else:
                # 假设拉普拉斯损失量级约为 0.1，自动给出一个合适权重。
                lap_fac = LAP_AUTO_SCALE * 0.1 / lap_loss.item()
            min_lap_fac = lap_fac * LAP_MIN_RATIO
        else:
            # 随迭代指数衰减到 ``min_lap_fac``，后期允许更多几何自由度。
            lap_fac = (
                (lap_fac - min_lap_fac) * 10 ** (-it * 1e-6) + min_lap_fac
            )

        total_loss = (
            img_loss + lap_loss * lap_fac + FLAGS.normal_weight * normal_loss
        )
        total_loss_vec.append(total_loss.item())

        # 4.4 反向传播与参数更新。
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        img_loss_vec.append(img_loss.detach().cpu().item())
        lap_loss_vec.append(lap_loss.detach().cpu().item())
        normal_loss_vec.append(normal_loss.detach().cpu().item())

        # 4.5 把贴图限制在合法范围，避免颜色/粗糙度越界。
        normal_map_opt.clamp_(min=-1, max=1)
        kd_map_opt.clamp_(min=0, max=1)
        ks_map_opt.clamp_rgb_(
            minR=0,
            maxR=1,
            minG=FLAGS.min_roughness,
            maxG=1.0,
            minB=0.0,
            maxB=1.0,
        )

        iter_dur_vec.append(time.time() - iter_start_time)
        learning_rate_vec.append(optimizer.param_groups[0]["lr"])

        # 4.6 控制台日志。
        if log_interval and (it % log_interval == 0):
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            lap_loss_avg = np.mean(np.asarray(lap_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            remaining = (FLAGS.iter - it) * iter_dur_avg
            print(
                "iter=%5d, img_loss=%.6f, lap_loss=%.6f, lr=%.5f, "
                "time=%.1f ms, rem=%s"
                % (
                    it,
                    img_loss_avg,
                    lap_loss_avg * lap_fac,
                    optimizer.param_groups[0]["lr"],
                    iter_dur_avg * 1000,
                    util.time_to_text(remaining),
                )
            )

        lap_fac_vec.append(lap_fac)

    # ------------------------------------------------------------------
    # 5. 导出曲线、日志和最终网格
    # ------------------------------------------------------------------
    _save_training_curves(
        out_dir,
        img_loss_vec,
        total_loss_vec,
        lap_fac_vec,
        learning_rate_vec,
        normal_loss_vec,
    )
    _save_loss_data(
        out_dir,
        img_loss_vec,
        total_loss_vec,
        lap_loss_vec,
        learning_rate_vec,
        normal_loss_vec,
        lap_fac_vec,
        iter_dur_vec,
    )
    obj.write_obj(os.path.join(out_dir, "mesh/"), opt_base_mesh.eval())


#----------------------------------------------------------------------------
# 命令行 / 配置
#----------------------------------------------------------------------------

# 与 argparse 参数同名；JSON 配置里可以只写需要覆盖的字段。
DEFAULT_FLAGS = {
    # 训练
    "iter": 5000,
    "batch": 1,
    "spp": 1,
    "layers": 1,
    "train_res": 512,
    "random_train_res": False,
    "display_res": None,
    "texture_res": [1024, 1024],
    "display_interval": 0,
    "save_interval": 1000,
    "learning_rate": None,
    "light_power": 5.0,
    "min_roughness": 0.08,
    "subdivision": 0,
    "custom_mip": False,
    "random_textures": False,
    "laplacian_factor": None,
    "relative_laplacian": False,
    "normal_weight": 0.01,
    "background": "white",
    "loss": "logl1",
    # 输入 / 输出
    "out_dir": None,
    "camera_eye": [0.0, 0.0, RADIUS],
    "camera_up": [0.0, 1.0, 0.0],
    "skip_train": [],
    "displacement": 0.15,
    "mtl_override": None,
    "ref_mesh": None,
    "base_mesh": None,
    "base_mtl": None,
    "config": None,
}


def build_parser():
    """构造命令行参数；所有参数都可被 JSON 配置覆盖。"""
    parser = argparse.ArgumentParser(
        description="BMS-DR: Building mesh simplification with differentiable rendering",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("-i", "--iter", type=int, help="optimization iterations")
    parser.add_argument("-b", "--batch", type=int, help="random views per iteration")
    parser.add_argument("-s", "--spp", type=int, help="samples per pixel")
    parser.add_argument("-l", "--layers", type=int, help="number of texture layers")
    parser.add_argument("-r", "--train-res", type=int, help="training resolution")
    parser.add_argument(
        "-rtr", "--random-train-res", action="store_true",
        help="randomize the training resolution"
    )
    parser.add_argument("-dr", "--display-res", type=int, help="display resolution")
    parser.add_argument(
        "-tr", "--texture-res", nargs=2, type=int,
        help="texture resolution, e.g. -tr 1024 1024"
    )
    parser.add_argument(
        "-di", "--display-interval", type=int, help="display interval"
    )
    parser.add_argument("-si", "--save-interval", type=int, help="image save interval")
    parser.add_argument("-lr", "--learning-rate", type=float)
    parser.add_argument("-lp", "--light-power", type=float)
    parser.add_argument("-mr", "--min-roughness", type=float)
    parser.add_argument("-sd", "--subdivision", type=int)
    parser.add_argument("-mip", "--custom-mip", action="store_true")
    parser.add_argument("-rt", "--random-textures", action="store_true")
    parser.add_argument("-lf", "--laplacian-factor", type=float)
    parser.add_argument("-rl", "--relative-laplacian", action="store_true")
    parser.add_argument(
        "--normal-weight", type=float, help="weight of the face normal loss"
    )
    parser.add_argument("-bg", "--background", choices=["black", "white", "checker"])
    parser.add_argument(
        "--loss", choices=["logl1", "logl2", "mse", "smape", "relativel2"]
    )
    parser.add_argument("-o", "--out-dir", type=str, help="output directory")
    parser.add_argument(
        "--config", type=str, help="JSON config file (CLI flags override it)"
    )
    parser.add_argument(
        "-rm", "--ref-mesh", "--ref_mesh", dest="ref_mesh",
        help="reference mesh to fit"
    )
    parser.add_argument(
        "-bm", "--base-mesh", "--base_mesh", dest="base_mesh",
        help="initial mesh; skips QEM simplification and UV generation"
    )
    parser.add_argument(
        "--base-mtl", "--base_mtl", dest="base_mtl",
        help="material override for --base-mesh"
    )
    parser.add_argument(
        "--qem-exe", dest="qem_exe", default=None,
        help="override the QEM executable (default: bin/qem/qem.exe)"
    )
    return parser


def load_flags(args):
    """合并默认值、JSON 配置和显式命令行参数。

    优先级：默认值 < JSON 配置 < 命令行参数。使用
    ``argument_default=SUPPRESS`` 后，未显式传入的参数不会出现在
    ``args`` 里，因此不会覆盖 JSON 中的同名配置。
    """
    flags = argparse.Namespace(**DEFAULT_FLAGS)

    if getattr(args, "config", None):
        with open(args.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        flags.__dict__.update(config)

    flags.__dict__.update(vars(args))

    # display_res 未单独指定时跟随训练分辨率。
    if flags.display_res is None:
        flags.display_res = flags.train_res
    return flags


def default_model_name(ref_mesh):
    """用模型所在文件夹名（如 ``0001``）作为默认输出目录名。"""
    path = Path(ref_mesh)
    parent = path.parent.name
    project_root = Path(__file__).resolve().parent.name
    if parent and parent != project_root:
        return parent
    return path.stem


def prepare_base_mesh(FLAGS):
    """返回初始低模及其材质路径。

    优先级：
    1. ``--base-mesh`` 显式提供（跳过 QEM 和 UV）；
    2. 否则对参考模型执行 QEM 简化，再用内置 UV 展开生成低模。
    """
    if FLAGS.base_mesh:
        base_mesh = str(Path(FLAGS.base_mesh).expanduser().resolve())
        if not Path(base_mesh).is_file():
            raise IOError("base mesh not found: %s" % base_mesh)
        base_mtl = FLAGS.base_mtl
        if base_mtl:
            base_mtl = str(Path(base_mtl).expanduser().resolve())
        print("Using provided base mesh: %s" % base_mesh)
        return base_mesh, base_mtl

    print("Simplifying reference mesh with QEM ...")
    status, simplified = qem_simplify(
        FLAGS.ref_mesh, FLAGS.out_dir, exe=getattr(FLAGS, "qem_exe", None)
    )
    if int(status) != 0:
        raise RuntimeError("QEM simplification failed with exit code %s" % status)

    print("Generating UV coordinates ...")
    return generate_uv(simplified, FLAGS.out_dir)


def main():
    """命令行入口：校验输入 -> 准备低模 -> 保存配置 -> 开始优化。"""
    parser = build_parser()
    args = parser.parse_args()
    FLAGS = load_flags(args)

    if not FLAGS.ref_mesh:
        parser.error(
            "a reference mesh is required: pass --ref-mesh/-rm or set "
            "'ref_mesh' in --config"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "BMS-DR requires a CUDA-capable GPU and a CUDA build of PyTorch"
        )
    torch.cuda.empty_cache()

    # 参考模型路径必须是绝对路径，后续 QEM / UV / 日志都基于它。
    FLAGS.ref_mesh = str(Path(FLAGS.ref_mesh).expanduser().resolve())
    if not Path(FLAGS.ref_mesh).is_file():
        parser.error("reference mesh not found: %s" % FLAGS.ref_mesh)

    # 未指定输出目录时使用 out/<模型目录名>。
    out_dir = FLAGS.out_dir or os.path.join(
        "out", default_model_name(FLAGS.ref_mesh)
    )
    FLAGS.out_dir = str(Path(out_dir).expanduser().resolve())
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    print("Output directory:", FLAGS.out_dir)

    # CGAL 读不了 OBJ 里的 ``l`` 线段记录，先清理。
    check_obj_file(FLAGS.ref_mesh)

    FLAGS.base_mesh, FLAGS.base_mtl = prepare_base_mesh(FLAGS)
    if FLAGS.base_mtl:
        FLAGS.base_mtl = str(FLAGS.base_mtl)

    # 保存本次运行的完整配置，方便复现实验。
    config_path = os.path.join(FLAGS.out_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            vars(FLAGS),
            handle,
            indent=4,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    print("Config saved to:", config_path)

    optimize_mesh(FLAGS, FLAGS.out_dir)
    return 0


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
