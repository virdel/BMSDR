import os
import pymeshlab
import trimesh
import xatlas



def uv_generate(input,output_dir):
    input = os.path.abspath(input)
    print("uv generate input",input)
    output=os.path.join(output_dir,"mesh_alpha_uv.obj")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input)
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
    ms.save_current_mesh(output)
    return output,output+".mtl"


def uv_generate_xatlas(input,output_dir):
    '''
    存在部分法线反向
    Args:
        input:
        output_dir:

    Returns:

    '''
    # We use trimesh (https://github.com/mikedh/trimesh) to load a mesh but you can use any library.
    mesh = trimesh.load_mesh(input)
    # output = os.path.join(output_dir, "mesh_alpha_uv.obj")
    # The parametrization potentially duplicates vertices.
    # `vmapping` contains the original vertex index for each new vertex (shape N, type uint32).
    # `indices` contains the vertex indices of the new triangles (shape Fx3, type uint32)
    # `uvs` contains texture coordinates of the new vertices (shape Nx2, type float32)
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    print(vmapping, indices, uvs)
    print(len(vmapping))
    # Trimesh needs a material to export uv coordinates and always creates a *.mtl file.
    # Alternatively, we can use the `export` helper function to export the mesh as obj.
    xatlas.export("test.obj", mesh.vertices[vmapping], indices, uvs)
    # material = trimesh.visual.texture.SimpleMaterial()
    # # material=trimesh.visual.material.empty_material()
    # color_visuals = trimesh.visual.TextureVisuals(uv=uvs,material=material)
    # mesh = trimesh.Trimesh(vertices=mesh.vertices[vmapping], faces=indices, visual=color_visuals, validate=True, process=False)
    # trimesh.repair.fix_normals(mesh,True)
    # mesh.export("test.obj",file_type="obj")
    # Both `xatlas.parametrize` and `xatlas.export` also accept vertex normals

def blender_uv_generating(input,output_dir):
    output = os.path.join(output_dir, "mesh_alpha_uv.obj")
    mtl= os.path.join(output_dir, "mesh_alpha_uv.mtl")
    blender_path=r"D:\program\blender-3.3.0-windows-x64\blender.exe"
    blender_python=r"E:\code\nvdiffmodeling-main\python_blender.py"
    blender_cmd="{} -b -P {} -- {} {}".format(blender_path,blender_python,input,output)
    os.system(blender_cmd)
    return output,mtl


if __name__=="__main__":
    input=r"E:\code\nvdiffmodeling-main\experiment\result\out\alpha_shape_uvpacker\0001\mesh\mesh.obj"
    output=r"E:/"
    uv_generate_xatlas(input, output)
    ##blender_uv_generating(input, output)
    # print(uv_generate(input,output))