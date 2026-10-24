
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


def check_obj_file(filename):
    '''
    用于处理输入obj文件，使得CGAL可以顺利读入
    Args:
       filename:

    Returns:

    '''
    try:
        with open(filename, encoding="utf-8") as f:
            lines = f.readlines()  ## 读取每一行
    except:
        with open(filename,encoding="ANSI") as f:
            lines = f.readlines()  ## 读取每一行

    new_lines=[line for line in lines if line[:2]!="l "]
    with open(filename, encoding="utf-8",mode="w") as f:
        for line in new_lines:
            f.write(line)
            # f.writelines("\n")


if __name__=="__main__":
    file_name=r"E:\code\nvdiffmodeling-main\models\models\0063\mesh.obj"
    # check_obj_file(file_name)
    get_vertex_triangle_num(file_name)
