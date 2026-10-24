## 为数据集中每个模型产生10个视点
import numpy as np
from pathlib import Path

def generated_random_point_from_sphere(radius=2.0):

    u,v=np.random.rand(2)
    phi=2*np.pi*u
    theta=np.arccos(2*v-1)
    x=radius*np.sin(theta)*np.cos(phi)
    y= radius*np.sin(theta) * np.sin(phi)
    z =radius*np.cos(theta)


    return np.array([x,y,z])


sample_number=10
ref_models_path=r"E:\code\nvdiffmodeling-main\模型"
path = Path(ref_models_path)
files = [i for i in path.iterdir() ]

print(files)

for i in files:
    points=[generated_random_point_from_sphere() for i in range(sample_number)]
    np.save(str(i.joinpath("points.npy")),np.stack(points))

