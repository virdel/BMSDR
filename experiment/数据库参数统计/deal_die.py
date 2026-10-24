##  处理三维模型文件

from pathlib  import  Path
import shutil
# data_dir=r"E:\download\OBJ_MODELS\OBJ_MODELS"
# target_dir=r"E:\download\OBJ_MODELS\OBJ_MODELS_target"
data_dir=r"E:\download\OBJ_MODELS\OBJ_MODELS"
target_dir=r"E:\download\OBJ_MODELS\OBJ_MODELS_target"

data_dir=Path(data_dir)
target_dir=Path(target_dir)



if not target_dir.exists():
    target_dir.mkdir()


obj_list=[f for f in data_dir.iterdir() if f.is_file() and f.suffix==".obj" ]

# print([str(f) for f in data_dir.iterdir() if f.is_file() and f.suffix==".obj" ])
# print([str(f.stem) for f in data_dir.iterdir() if f.is_file() and f.suffix==".obj" ])
# print([str(f.parent) for f in data_dir.iterdir() if f.is_file() and f.suffix==".obj" ])
# print(len(obj_list))
for i in obj_list:
    print(i.stem)
    dir=target_dir/i.stem
    if not dir.exists():
        dir.mkdir()

    mtl_path=data_dir/(str(i.stem)+".mtl")

    shutil.copy(i,dir)
    shutil.copy(mtl_path,dir)

    print(mtl_path)
    source_mtl_dir=data_dir/i.stem
    target_mtl_dir=dir/i.stem

    print(source_mtl_dir)
    print(target_mtl_dir)

    if source_mtl_dir.exists():
        shutil.copytree(source_mtl_dir,target_mtl_dir)



    # shutil.copy(str(i),target_dir)


