from pathlib import Path
import  sys
import pandas as pd

path=r"E:\data\三维模型库"

path = Path(path)

obj_files = [i for i in path.glob(f"**\*.obj")]
mtl_files = [i for i in path.glob(f"**\*.mtl")]

print(obj_files)
print(mtl_files)

print(len(obj_files))
print(len(mtl_files))

print([ i.parent.stem for  i in obj_files])

# for i,j in enumerate(obj_files):
#     print(i)
#     print(j)


df=pd.DataFrame()


df["model_name"]=[ i.parent.stem for  i in obj_files]

print(df)


df.to_csv("model_name.csv",encoding="utf-8")
# writer = pd.ExcelWriter('model_name.xlsx')
#
# df.to_excel(writer)
#
# writer.save()

