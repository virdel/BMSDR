import pymeshlab
import pandas as pd

df=pd.read_csv("../data.csv",encoding="utf-8")

print(df)
vertex=[]
triangles=[]


for row_index,row in df.iterrows():
    print(row)
    if row["model_name"]=="0023民众乐园":
        vertex.append(0)
        triangles.append(0)
        continue

    ms = pymeshlab.MeshSet()
    model_path=row["model_path"]

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(model_path)

    mesh = ms.current_mesh()

    vertex.append(mesh.vertex_number())
    triangles.append(mesh.face_number())
    print(mesh.vertex_number())
    print(mesh.face_number())

    ms.clear()




print(vertex)
print(triangles)

df["vertex"]=vertex
df["triangles"]=triangles

df.to_csv("data_vertex_triangles.csv",encoding="utf-8")