
from pathlib import Path
import  sys

print(sys.stdout.encoding)
print(sys.getdefaultencoding())

path=r"E:\data\三维模型库"

path = Path(path)

obj_files = [i for i in path.glob(f"**\*.obj")]
mtl_files = [i for i in path.glob(f"**\*.mtl")]

print(obj_files)
print(mtl_files)

print(len(obj_files))
print(len(mtl_files))


#
# for i in obj_files:
#     with open(str(i),encoding="utf-8") as f:
#         seq=f.read()
#     print(str(i))
#     print(seq)
#     # print(str(i).encode("gbk").decode("gbk"))
#     # print(seq)
#     #
#     # try:
#     #     with open(str(i),encoding="utf-8") as f:
#     #         seq=f.read()
#     # except:
#     #     with open(str(i),encoding="gbk") as f:
#     #         seq=f.read()
#     # finally:
#     #     with open(str(i),"w",encoding="utf-8") as f:
#     #         f.write(seq)
#     #     print(str(i))
#
# for i in mtl_files:
#     # try:
#     #     with open(str(i),encoding="utf-8") as f:
#     #         seq=f.read()
#     # except:
#     #     with open(str(i),encoding="gbk") as f:
#     #         seq=f.read()
#     # finally:
#     #     with open(str(i),"w",encoding="utf-8") as f:
#     #         f.write(seq)
#     #     print(str(i))
#
#     with open(str(i),encoding="utf-8") as f:
#         seq=f.read()
#     print(str(i))
#     print(seq)


with open(str(r"E:\data\三维模型库\0044英美烟草公司\044yancao.obj"),encoding="utf-8") as f:
        seq=f.readlines()

print(seq[3])

seq[3]="mtllib 044yancao.mtl"

with open("044yancao.obj","w",encoding="utf-8") as f:
    seq=f.writelines(seq)

