

import os

import pandas as pd

os.chdir(r"E:\code\nvdiffmodeling-main")
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import src.renderutils as ru
from src import obj
from src import util
from src import mesh
from src import texture
from src import render
from src import regularizer
from src.mesh import Mesh
import  matplotlib.pyplot as plt
import numpy as np
import  torch
import nvdiffrast.torch as dr
import  json

import matplotlib.pyplot as plt

import skimage
from pathlib import Path

import lpips


def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"

def compute_ssim(image1,image2):
    pass

def generated_random_point_from_sphere(radius=1.0):


    # x=np.random.randn()
    # y=np.random.rand()
    # z=np.random.randn()
    #
    # r=np.sqrt((x*x+y*y+z*z))
    #
    # x=x/r
    # y=y/r
    # z=z/r
    #
    # x=x*pow(radius,0.5)
    # y = y * pow(radius, 0.5)
    # z = z * pow(radius, 0.5)

    u,v=np.random.rand(2)
    phi=2*np.pi*u
    theta=np.arccos(2*v-1)
    x=radius*np.sin(theta)*np.cos(phi)
    y= radius*np.sin(theta) * np.sin(phi)
    z =radius*np.cos(theta)


    return np.array([x,y,z])

def get_model(ori_mesh,opt_mesh):

    RADIUS = 3.5
    proj_mtx = util.projection(x=0.4, f=1000.0)  ## 投影矩阵
    ori_mesh = load_mesh(ori_mesh)
    opt_mesh = load_mesh(opt_mesh)

    ori_tri_num = ori_mesh.t_pos_idx.shape[0]

    ## 归一化
    normalized_ori_mesh = mesh.unit_size(ori_mesh)
    normalized_opt_mesh = mesh.unit_size(opt_mesh)

    ## 切线空间
    render_ori_mesh = mesh.compute_tangents(normalized_ori_mesh)
    render_opt_mesh = mesh.compute_tangents(normalized_opt_mesh)

    ## aabb
    ori_mesh_aabb = mesh.aabb(render_ori_mesh.eval())
    opt_mesh_aabb = mesh.aabb(render_opt_mesh.eval())


    return render_ori_mesh,render_opt_mesh,ori_mesh_aabb,ori_tri_num


def get_image_pair(render_ori_mesh,render_opt_mesh,ori_mesh_aabb,eye_point):


    proj_mtx = util.projection(x=0.4, f=1000.0)  ## 投影矩阵
    eye = np.array([2.5, 0, -2.5])
    # eye=generated_random_point_from_sphere(4)
    eye=eye_point
    up = np.array([0, 1.0, 0.0])
    at = np.array([0, 0, 0])

    a_mv = util.lookAt(eye, at, up)
    a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]  ## 1*3*3

    a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
    a_campos = np.linalg.inv(a_mv)[None, :3, 3]

    params = {'mvp': a_mvp, 'lightpos': a_lightpos, 'campos': a_campos,
              'resolution': [1024, 1024],
              'time': 0}

    mesh_scale = 1.0
    ori_mesh = mesh.center_by_reference(render_ori_mesh.eval(params),ori_mesh_aabb, mesh_scale)
    opt_mesh = mesh.center_by_reference(render_opt_mesh.eval(params), ori_mesh_aabb, mesh_scale)

    light_power = 5.0
    spp = 1
    min_roughness = 0.08
    background = torch.ones((1, 1024, 1024, 3), dtype=torch.float32, device='cuda')
    glctx = dr.RasterizeGLContext()
    display_res = 1024


    ori_img ,_= render.render_mesh(glctx, ori_mesh, a_mvp, a_campos, a_lightpos, light_power, display_res,
                                 num_layers=1, spp=spp, background=background, min_roughness=min_roughness)

    opt_img,_=render.render_mesh(glctx, opt_mesh, a_mvp, a_campos, a_lightpos, light_power, display_res,
                                 num_layers=1, spp=spp, background=background, min_roughness=min_roughness)

    ori_img = util.scale_img_nhwc(ori_img, [display_res, display_res])
    opt_img = util.scale_img_nhwc(opt_img, [display_res, display_res])

    ori_img=util.tonemap_srgb(ori_img)
    opt_img = util.tonemap_srgb(opt_img)


    ori_img= ori_img.detach().cpu().numpy()
    opt_img = opt_img.detach().cpu().numpy()

    ori_img=np.squeeze(ori_img)
    opt_img = np.squeeze(opt_img)

    ori_img=np.clip(np.rint(ori_img * 255.0), 0, 255).astype(np.uint8)
    opt_img = np.clip(np.rint(opt_img * 255.0), 0, 255).astype(np.uint8)

    return ori_img,opt_img



#
# def get_image_pair(ori_mesh,opt_mesh):
#
#     print("ori_image:",ori_mesh)
#     print("opt_image:",opt_mesh)
#     RADIUS = 3.5
#     proj_mtx = util.projection(x=0.4, f=1000.0)  ## 投影矩阵
#     ori_mesh=load_mesh(ori_mesh)
#     opt_mesh = load_mesh(opt_mesh)
#
#     ori_tri_num=ori_mesh.t_pos_idx.shape[0]
#
#       ## 归一化
#     normalized_ori_mesh=mesh.unit_size(ori_mesh)
#     normalized_opt_mesh = mesh.unit_size(opt_mesh)
#
#     ## 切线空间
#     render_ori_mesh=mesh.compute_tangents(normalized_ori_mesh)
#     render_opt_mesh = mesh.compute_tangents(normalized_opt_mesh)
#
#
#     ## aabb
#     ori_mesh_aabb=mesh.aabb(render_ori_mesh.eval())
#     opt_mesh_aabb = mesh.aabb(render_opt_mesh.eval())
#
#
#
#     eye = np.array([2.5, 0, -2.5])
#     eye=generated_random_point_from_sphere(4)
#     up = np.array([0, 1.0, 0.0])
#     at = np.array([0, 0, 0])
#
#     a_mv = util.lookAt(eye, at, up)
#     a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]  ## 1*3*3
#
#     a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
#     a_campos = np.linalg.inv(a_mv)[None, :3, 3]
#
#     params = {'mvp': a_mvp, 'lightpos': a_lightpos, 'campos': a_campos,
#               'resolution': [1024, 1024],
#               'time': 0}
#
#     mesh_scale = 2.0
#     ori_mesh = mesh.center_by_reference(render_ori_mesh.eval(params),ori_mesh_aabb, mesh_scale)
#     opt_mesh = mesh.center_by_reference(render_opt_mesh.eval(params), ori_mesh_aabb, mesh_scale)
#
#     light_power = 5.0
#     spp = 1
#     min_roughness = 0.08
#     background = torch.ones((1, 1024, 1024, 3), dtype=torch.float32, device='cuda')
#     glctx = dr.RasterizeGLContext()
#     display_res = 1024
#
#
#     ori_img = render.render_mesh(glctx, ori_mesh, a_mvp, a_campos, a_lightpos, light_power, display_res,
#                                  num_layers=1, spp=spp, background=background, min_roughness=min_roughness)
#
#     opt_img=render.render_mesh(glctx, opt_mesh, a_mvp, a_campos, a_lightpos, light_power, display_res,
#                                  num_layers=1, spp=spp, background=background, min_roughness=min_roughness)
#
#     ori_img = util.scale_img_nhwc(ori_img, [display_res, display_res])
#     opt_img = util.scale_img_nhwc(opt_img, [display_res, display_res])
#
#     ori_img=util.tonemap_srgb(ori_img)
#     opt_img = util.tonemap_srgb(opt_img)
#
#
#     ori_img= ori_img.detach().cpu().numpy()
#     opt_img = opt_img.detach().cpu().numpy()
#
#     ori_img=np.squeeze(ori_img)
#     opt_img = np.squeeze(opt_img)
#
#     ori_img=np.clip(np.rint(ori_img * 255.0), 0, 255).astype(np.uint8)
#     opt_img = np.clip(np.rint(opt_img * 255.0), 0, 255).astype(np.uint8)
#
#     return ori_img,opt_img,ori_tri_num


def psnr_metric(ori_mesh,opt_mesh):
    RADIUS = 3.5
    proj_mtx = util.projection(x=0.4, f=1000.0)  ## 投影矩阵
    ori_mesh=load_mesh(ori_mesh)
    opt_mesh = load_mesh(opt_mesh)
      ## 归一化
    normalized_ori_mesh=mesh.unit_size(ori_mesh)
    normalized_opt_mesh = mesh.unit_size(opt_mesh)

    ## 切线空间
    render_ori_mesh=mesh.compute_tangents(normalized_ori_mesh)
    render_opt_mesh = mesh.compute_tangents(normalized_opt_mesh)


    ## aabb
    ori_mesh_aabb=mesh.aabb(render_ori_mesh.eval())

    opt_mesh_aabb = mesh.aabb(render_opt_mesh.eval())

    eye = np.array([1.0, 0, 0])
    eye=generated_random_point_from_sphere(4)
    up = np.array([0, 1.0, 0.0])
    at = np.array([0, 0, 0])

    a_mv = util.lookAt(eye, at, up)
    a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]  ## 1*3*3

    a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
    a_campos = np.linalg.inv(a_mv)[None, :3, 3]

    params = {'mvp': a_mvp, 'lightpos': a_lightpos, 'campos': a_campos,
              'resolution': [2048, 2048],
              'time': 0}

    mesh_scale = 1.0
    ori_mesh = mesh.center_by_reference(render_ori_mesh.eval(params),ori_mesh_aabb, mesh_scale)
    opt_mesh = mesh.center_by_reference(render_opt_mesh.eval(params), ori_mesh_aabb, mesh_scale)

    light_power = 5.0
    spp = 1
    min_roughness = 0.08
    background = torch.ones((1, 2048, 2048, 3), dtype=torch.float32, device='cuda')
    glctx = dr.RasterizeGLContext()
    display_res = 2048


    ori_img = render.render_mesh(glctx, ori_mesh, a_mvp, a_campos, a_lightpos, light_power, display_res,
                                 num_layers=1, spp=spp, background=background, min_roughness=min_roughness)

    opt_img=render.render_mesh(glctx, opt_mesh, a_mvp, a_campos, a_lightpos, light_power, display_res,
                                 num_layers=1, spp=spp, background=background, min_roughness=min_roughness)

    ori_img = util.scale_img_nhwc(ori_img, [display_res, display_res])
    opt_img = util.scale_img_nhwc(opt_img, [display_res, display_res])

    ori_img= ori_img.detach().cpu().numpy()
    opt_img = opt_img.detach().cpu().numpy()

    ori_img=np.squeeze(ori_img)
    opt_img = np.squeeze(opt_img)

    # ori_img=ori_img.astype(np.int32)
    # opt_img = opt_img.astype(np.int32)

    print(ori_img.shape)
    print(opt_img.shape)

    print(ori_img.dtype)
    print(ori_img.max())

    # print(skimage.metrics.peak_signal_noise_ratio(ori_img,opt_img))



    plt.axis("off")
    plt.imshow(ori_img) # approximate sRGB tonemapping
    plt.show()
    plt.axis("off")
    plt.imshow(opt_img) # approximate sRGB tonemapping
    plt.show()

def get_mse_metric(a,b):
    return skimage.metrics.mean_squared_error(a,b)
def get_psnr_metric(a,b):
    return skimage.metrics.peak_signal_noise_ratio(a,b)

def get_ssim_metric(a,b):
    return skimage.metrics.structural_similarity(a,b,channel_axis=2)
    # return skimage.metrics.structural_similarity(a,b)
def get_LPIPS_metric(a,b):
    loss_fn_vgg=lpips.LPIPS(net="vgg")
    a=lpips.im2tensor(a)
    b=lpips.im2tensor(b)
    # print(a.shape)
    with torch.no_grad():
        score=loss_fn_vgg(a, b, normalize=True)
    return score.numpy().item()


def get_all_metrics():

    ref_models_path=r"E:\code\nvdiffmodeling-main\模型"

    target_models_path=r"E:\code\nvdiffmodeling-main\out一步生成\tri_10000_0.05"

    path = Path(ref_models_path)
    files = [i for i in path.glob(f"**\*.obj")]

    name="tri_10000_0.05"

    image_save_path = "./指标/"+name

    os.makedirs(image_save_path,exist_ok=True)

    data=[]

    for file in files:
        target_model_path=Path(target_models_path,file.parent.stem,"mesh","mesh.obj")

        # print(str(file))
        # target_model_path = Path(target_models_path, file.parent.stem,  "mesh.obj")
        if not target_model_path.exists():
            continue
        ## 导入球面上采样点
        points = np.load(str(Path(file.parent,"points.npy")))
        render_ori_mesh,render_opt_mesh,ori_mesh_aabb,ori_tri_num = get_model(str(file),str(target_model_path))

        # print("ori_mesh_aabb",ori_mesh_aabb)
        for i,point in enumerate(list(points)):
            ## 得到两张渲染图像
            image1,image2=get_image_pair(render_ori_mesh,render_opt_mesh,ori_mesh_aabb,point) ## GT test

            # plt.axis("off")
            # plt.imshow(np.concatenate((image1,image2)))  # approximate sRGB tonemapping
            # plt.savefig(os.path.join(image_save_path,str(file.parent.stem)+str(i)+".png"))

            # plt.axis("off")
            # plt.imshow(image2)  # approximate sRGB tonemapping
            # plt.show()
            # print(ori_tri_num)
            # mse=get_mse_metric(image1,image2)
            psnr=get_psnr_metric(image1,image2)
            ssim=get_ssim_metric(image1,image2)
            lpips_score=get_LPIPS_metric(image1,image2)
            # print("lpips_score",lpips_score)
            data.append((str(file.parent.stem)+str(i),psnr,ssim,lpips_score, ori_tri_num))


    df=pd.DataFrame(data,columns=["name","psnr","ssim","lpips_score","origin_tris"])
    df.to_csv("./指标/{}.csv".format(name))

a=r"E:\code\nvdiffmodeling-main\out一步生成\tri_3000\0001八路军驻武汉办事处\mesh\mesh.obj"
b=r"E:\code\nvdiffmodeling-main\模型\0001八路军驻武汉办事处\mesh.obj"

# psnr_metric(a,b)

get_all_metrics()