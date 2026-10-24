import  matplotlib.pyplot as plt
import  pandas as pd




df=pd.read_csv("data_vertex_triangles.csv",encoding="utf-8")

print(df)

fig,ax = plt.subplots(1,1)
a=df["vertex"].tolist()
b=df["triangles"].tolist()
# ax.hist(a,bins=50)
# ax.set_title("histogram of triangles")
#
# ax.set_xlabel('marks')
# ax.set_ylabel('#numbers')
# plt.show()

ax.scatter(a,b)
ax.set_title("histogram of vertex")

ax.set_xlabel('vertex')
ax.set_ylabel('triangles')
plt.show()
#
# ax.hist(a[50:],bins=50)
# ax.set_title("histogram of triangles")
#
# ax.set_xlabel('marks')
# ax.set_ylabel('#numbers')
# plt.show()