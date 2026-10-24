
import os
from src import obj
from src import mesh


import os
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
mesh_path=r"E:\download\OBJ_MODELS\OBJ_MODELS\RELIGIOUSchurch_mesh2017.obj"
ref_mesh = load_mesh(mesh_path)


ref_mesh=mesh.unit_size(ref_mesh)

obj.write_obj("./", ref_mesh.eval())