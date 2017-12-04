import os
import shutil

train_root = 'data/sun_train'
val_root = 'data/sun_val'

train_txt = "/home/sunqc/DR/data/train.txt"
val_txt = "/home/sunqc/DR/data/valid.txt"


if __name__ == "__main__":
    # make directory
    if os.path.exists(train_root): shutil.rmtree(train_root)
    if os.path.exists(val_root): shutil.rmtree(val_root)
    os.makedirs(train_root+"/0")
    os.makedirs(train_root + "/1")
    os.makedirs(val_root+"/0")
    os.makedirs(val_root + "/1")

    # move train data
    global_img_idx = 0
    with open(train_txt, "rb") as f:
        lines = f.readlines()
    for line in lines:
        path, label = line.strip().split(" ")
        if label == "0":
            shutil.move(path, train_root+"/0/"+str(global_img_idx)+"_"+os.path.basename(path))
        elif label == "1":
            shutil.move(path, train_root + "/1/" + str(global_img_idx) + "_" + os.path.basename(path))
        else:
            raise ValueError("label error, label="+label)
        global_img_idx += 1

    # move val data
    with open(val_txt, "rb") as f:
        lines = f.readlines()
    for line in lines:
        path, label = line.strip().split(" ")
        if label == "0":
            shutil.move(path, val_root+"/0/"+str(global_img_idx)+"_"+os.path.basename(path))
        elif label == "1":
            shutil.move(path, val_root+"/1/"+str(global_img_idx)+"_"+os.path.basename(path))
        else:
            raise ValueError("label error, label=" + label)
        global_img_idx += 1

