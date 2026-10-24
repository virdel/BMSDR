import  os
import pymeshlab
from pathlib import Path
import pandas as pd

from lfd import LightFieldDistance
import trimesh
import subprocess

def get_vertex_triangle_num(filename):
    try:
        with open(filename, encoding="utf-8") as f:
            lines = f.readlines()  ## 读取每一行
    except:
        with open(filename,encoding="ANSI") as f:
            lines = f.readlines()  ## 读取每一行
    tri_numbers=0
    vertex_numbers=0
    for line in lines:
        if(line[:2]=="v "):
            vertex_numbers+=1
        if (line[0] == "f"):
            tri_numbers+=1
    return vertex_numbers,tri_numbers

def get_vertex_tri_numbers(model_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(model_path)
    mesh = ms.current_mesh()
    vertex=mesh.vertex_number()
    triangles=mesh.face_number()
    return vertex,triangles

def get_topological_measure(model_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(model_path)
    mesh = ms.current_mesh()

    measure=ms.get_topological_measures()
    measure["face_number"]=mesh.face_number()

    measure["diagonal"]=mesh.bounding_box().diagonal()

    return measure




def get_one_model_visual_metric(model1,model2):

    cmd = r"E:\code\nvdiffmodeling-main\指标\binary\visual-metric.exe " + model1 + " " + model2
    cmd=cmd.replace("\\","/")

    print(cmd)
    string = os.popen(cmd).readlines()
    print(string)
    return float(string[-1].split(":")[-1].strip())

def get_one_model_new_visual_metric(model1,model2):

    cmd = r"E:\code\nvdiffmodeling-main\指标\metric_bin\learn_opengl.exe " + model1 + " " + model2
    cmd=cmd.replace("\\","/")

    print(cmd)
    # string = os.popen(cmd).readlines()
    # print(string)

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    string=result.stdout.decode("utf-8").split()
    print(string)
    # print(result.stderr)


    return float(string[-1].strip())

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def get_one_model_LFD(model1,model2):
# rest of codemesh_1:trimesh.Trimesh=...mesh_2:trimesh.Trimesh=...lfd_value:float=LightFieldDistance(verbose=True).get_distance(mesh_1.vertices,mesh_1.faces,mesh_2.vertices,mesh_2.faces)
    mesh_1=trimesh.load_mesh(model1)
    mesh_2=trimesh.load_mesh(model2)

    mesh_1=as_mesh(mesh_1)
    mesh_2=as_mesh(mesh_2)
    lfd_value=LightFieldDistance(verbose=True).get_distance(mesh_1.vertices,mesh_1.faces,mesh_2.vertices,mesh_2.faces)
    print("lfd value:",lfd_value)
    return  lfd_value

def main(ref_model_path,target_model_path):

    path = Path(ref_model_path)
    files = [i for i in path.glob(f"**\*.obj")]
    model_name_vec=[]
    visual_metric_vec=[]
    ref_triangels=[]
    opt_triangels = []
    ref_vertexs=[]
    opt_vertexs=[]
    simply_ratios=[]
    # visual_func=get_one_model_new_visual_metric
    # visual_func =get_one_model_LFD
    visual_func =get_one_model_visual_metric
    for i in files:
        print(str(i))
        model_name=str(i.parent.stem)
        model_name_vec.append(model_name)
        # target_model=os.path.join(target_model_path,model_name,"mesh","mesh.obj")
        target_model = os.path.join(target_model_path, model_name,  "mesh.obj")
        # target_model = os.path.join(target_model_path, model_name+".obj")
        print(target_model)
        try:
            if os.path.exists(target_model):
                visual_metric_vec.append( visual_func(str(i),str(target_model)))
            else:
                visual_metric_vec.append(None)
        except:
            visual_metric_vec.append(None)

        if os.path.exists(target_model):
            opt = get_vertex_triangle_num(target_model)
            opt_vertexs.append(opt[0])
            opt_triangels.append(opt[1])
            try:
                ref=get_vertex_triangle_num(str(i))
                ref_vertexs.append(ref[0])
                ref_triangels.append(ref[1])
                simply_ratios.append(opt[1]/ref[1])
            except:
                ref_vertexs.append(None)
                ref_triangels.append(None)
                simply_ratios.append(None)
        else:
            opt_vertexs.append(None)
            opt_triangels.append(None)
            ref_vertexs.append(None)
            ref_triangels.append(None)
            simply_ratios.append(None)


    df=pd.DataFrame({"model_name":model_name_vec,"visual_metric":visual_metric_vec,"ref_vertexs":ref_vertexs,"ref_triangles":ref_triangels,"opt_vertexs":opt_vertexs,"opt_triangles":opt_triangels,"simply_ratios":simply_ratios})

    # df.to_csv("visual_metric_qem_0.01_blender.csv")
    # df.to_csv("visual_metric_alpha_shape_lr0.001_allmodel.csv")
    # df.to_csv("new_visual_metric_alpha_shape.csv")
    # df.to_csv("new_visual_metric_nvdiff.csv")
    # df.to_csv("visual_metric_nvdiffmodeling.csv")
    # df.to_csv("visual_metric_alpha_shape.csv")
    # df.to_csv("lfd_low_poly.csv")

    df.to_csv("visual_metric_simply3000.csv")

def change_name():

    path=r"E:\code\nvdiffmodeling-main\low_poly"
    path = Path(path)

    file_list = [f for f in Path(path).iterdir() ]
    for i in file_list:
        name=str(i.stem).split("-")[0]
        name="{:0>4d}.obj".format(int(name))
        print(str(path/name))
        i.rename(path/name)



def main_(ref_model_path):
    path = Path(ref_model_path)
    files = [i for i in path.glob(f"**\*.obj")]
    measure_list=[]
    for i in files:
        measure=get_topological_measure(str(i))
        measure_list.append(measure)
    df=pd.DataFrame(measure_list)
    df.to_csv("dataset_topo.csv")


def get_data_info():
    path=r"E:\code\nvdiffmodeling-main\simplygon_model\ratio_0.01"
    path = Path(path)
    files = [i for i in path.glob(f"**\*.obj")]
    info_list=[]
    for i in files:
        item={}
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(i))
        mesh = ms.current_mesh()
        item["face_number"]=mesh.face_number()
        item["name"]=str(i)
        info_list.append(item)

    df = pd.DataFrame(info_list)
    df.to_csv("simplygon_face_number.csv")






if __name__=="__main__":
    ref_model_path = r"E:\code\nvdiffmodeling-main\models"

    # target_model_path = r"E:\code\nvdiffmodeling-main\QEM_model\ratio_0.01"
    # target_model_path = r"E:\code\nvdiffmodeling-main\out\alpha_shape_lr0.001_allmodel_em_normalloss0.01_lapauto"
    # target_model_path=r"E:\code\nvdiffmodeling-main\low_poly"
    # target_model_path = r"E:\code\nvdiffmodeling-main\out\nvdiffmodeding"
    target_model_path =r"E:\code\nvdiffmodeling-main\simplygon_model\3000"
    main(ref_model_path,target_model_path)
    # change_name()


    # model1=r"E:\code\nvdiffmodeling-main\models\0001\mesh.obj"
    # model2=r"E:\code\nvdiffmodeling-main\out\alpha_shape_lr0.001_allmodel_em_normalloss0.01_lapauto\0001\mesh\mesh.obj"
    #
    # get_one_model_LFD(model1,model2)
    # main_(ref_model_path)

    # get_data_info()
