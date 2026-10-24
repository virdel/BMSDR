
import os
os.chdir(r"E:\code\nvdiffmodeling-main")
print(os.getcwd())
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

def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"

RADIUS = 3.5

proj_mtx = util.projection(x=0.4, f=1000.0)  ## 投影矩阵

print(proj_mtx)
mesh_path=r"E:\data\三维模型库\0160亭子1\mesh.obj"
ref_mesh = load_mesh(mesh_path)

# json_path=r"E:\download\BuildingNet_dataset_v0_1-20230421T031419Z-002\BuildingNet_dataset_v0_1\model_data\obj\face_labels\face_labels\COMMERCIALcastle_mesh0365.json"
#
#
# with open(json_path,"r",encoding="utf-8") as f:
#     content=json.load(f)
# index=set(content.values())
# index=dict.fromkeys(index,0)

# label_list=[]
#
# for i in content:
#     index[content[i]]+=1
#     if content[i]==0:
#         label_list.append(eval(i))

# print(index)

print(ref_mesh.t_pos_idx.shape)
print(ref_mesh.t_nrm_idx.shape)
print(ref_mesh.t_tex_idx.shape)

#
# ref_mesh.t_pos_idx=ref_mesh.t_pos_idx[label_list]
# ref_mesh.t_nrm_idx=ref_mesh.t_nrm_idx[label_list]
# ref_mesh.t_tex_idx=ref_mesh.t_tex_idx[label_list]
# # print(content)


print(ref_mesh.t_pos_idx.shape)


ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
# ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())

normalized_ref_mesh = mesh.unit_size(ref_mesh)

render_ref_mesh = mesh.compute_tangents(normalized_ref_mesh)

ref_mesh_aabb = mesh.aabb(render_ref_mesh.eval())

print("ref_mesh_aabb",ref_mesh_aabb)
print("normalized_ref_mesh",mesh.aabb(normalized_ref_mesh.eval()))

# background = torch.tensor(util.checkerboard(1024, 8), dtype=torch.float32, device='cuda')
background = torch.ones((1, 1024, 1024, 3), dtype=torch.float32, device='cuda')

eye=np.array([1.50, 0.00, 1.50 ])
up = np.array([ 0, 1.0, 0.0 ]) 
at = np.array([0, 0, 0])
a_mv = util.lookAt(eye, at, up)
a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]  ## 1*3*3
a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
a_campos = np.linalg.inv(a_mv)[None, :3, 3]

glctx = dr.RasterizeGLContext()
glctx = dr.RasterizeCudaContext("cuda")

params = {'mvp': a_mvp, 'lightpos': a_lightpos, 'campos': a_campos,
          'resolution': [1024, 1024],
          'time': 0}

mesh_scale=1.5
_opt_ref  = mesh.center_by_reference(render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale)




light_power=5.0
display_res=1024
spp=2
min_roughness=0.08


img_ref,_= render.render_mesh(glctx, _opt_ref, a_mvp, a_campos, a_lightpos, light_power, display_res,
                    num_layers=1, spp=spp, background=background, min_roughness=min_roughness)

img_ref  = util.scale_img_nhwc(img_ref,  [display_res,display_res])

img_ref=util.tonemap_srgb(img_ref)
print(img_ref.shape)

img_ref=img_ref.detach().cpu().numpy()


obj.write_obj("./",_opt_ref)

import matplotlib.pyplot as plt

plt.axis("off")
plt.imshow(img_ref[0]) # approximate sRGB tonemapping
plt.show()