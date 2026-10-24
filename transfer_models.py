from pathlib import  Path



def main():
    path=Path(r"E:\code\nvdiffmodeling-main\models")

    file_list=[f for f in Path(path).iterdir()]
    print(file_list)

    for i in file_list:
        new_path=i.parent/str(i.stem)[:4]
        # print(new_path)
        i.rename(new_path)

if __name__=="__main__":
    main()