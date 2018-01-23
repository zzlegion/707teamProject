import os
import shutil
import numpy as np
GAN_DIR = "/Users/liujianwei/OneDrive - inc/CMU_2017FALL/deep_learning_10707/homework/707TeamProject/data/gan"
DES_DIR = "/Users/liujianwei/OneDrive - inc/CMU_2017FALL/deep_learning_10707/homework/707TeamProject/data/sundata/all/train/0"

fs = []
for file_name in os.listdir(GAN_DIR):
    fs.append(file_name)
np.random.shuffle(fs)
for i in range(40):
    if os.path.exists(DES_DIR+"/"+fs[i]):
        pass
    else:
        shutil.copy(GAN_DIR+"/"+fs[i], DES_DIR+"/"+fs[i])
        print "copy" + fs[i] + "to " + DES_DIR
