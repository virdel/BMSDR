import  os

from pathlib import Path

import pandas as pd


def get_alpha_shape(input,output):
    cmd = r".\bin\alpha_shape\CGAL_learn.exe " + input + " " + output
    return_status=os.system(cmd)
    # result_string = os.popen(cmd).readlines()
    # print(result_string)
    print(return_status)
    return return_status,os.path.join(output,"mesh_50_600.obj")
def get_QEM_shape(input,output):
    cmd = r".\bin\qem\qem.exe " + input + " " + output
    return_status=os.system(cmd)
    # result_string = os.popen(cmd).readlines()
    # print(result_string)
    print(return_status)
    return return_status,os.path.join(output,"mesh_qem_3000.obj")





if __name__=="__main__":
    get_alpha_shape("","")