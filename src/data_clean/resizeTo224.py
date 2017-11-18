# encoding: utf-8
import cv2
import numpy as np
import os
import shutil
# the root holding the images
RESIZE_ROOT = "/Users/liujianwei/OneDrive - inc/CMU_2017FALL/deep_learning_10707/homework/707data/data/train"


# expand file_path img to be 224
def resize_to_224(file_path):
    img = cv2.imread(file_path)
    new_img = np.zeros(shape=(224, 224, 3))
    new_img[:img.shape[0], :img.shape[1], 0] = img[:, :, 0]
    new_img[:img.shape[0], :img.shape[1], 1] = img[:, :, 1]
    new_img[:img.shape[0], :img.shape[1], 2] = img[:, :, 2]
    cv2.imwrite(file_path, new_img)


# expand all the image in dir to be 224
def resize_all_to_224(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.splitext(file_path)[1].lower() in [".jpg", ".jpeg", ".png"]:
                resize_to_224(file_path)
            else:
                os.remove(file_path)


if __name__ == "__main__":
    resize_all_to_224(RESIZE_ROOT)
