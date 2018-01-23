# encoding: utf-8
import cv2
import numpy as np
import os
import shutil
# the root holding the images
RESIZE_ROOT = "/Users/liujianwei/OneDrive - inc/CMU_2017FALL/deep_learning_10707/homework/707data/data/data"


# expand file_path img to be 224
def pad_to_224(file_path):
    img = cv2.imread(file_path)
    new_img = np.zeros(shape=(224, 224, 3))
    new_img[:img.shape[0], :img.shape[1], 0] = img[:, :, 0]
    new_img[:img.shape[0], :img.shape[1], 1] = img[:, :, 1]
    new_img[:img.shape[0], :img.shape[1], 2] = img[:, :, 2]
    cv2.imwrite(file_path, new_img)


# expand file_path img to be 224
def resize_to_224(file_path):
    img = cv2.imread(file_path)
    new_img = cv2.resize(img, (224, 224))
    cv2.imwrite(file_path, new_img)


# resize image to any size
def resize_to_any(file_path, new_size, new_file_path):
    img = cv2.imread(file_path)
    new_img = cv2.resize(img, (new_size, new_size))
    if not new_file_path:
        new_file_path = file_path
    cv2.imwrite(new_file_path, new_img)


# expand all the image in dir to be 224
def resize_all_to_224(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.splitext(file_path)[1].lower() in [".jpg", ".jpeg", ".png"]:
                resize_to_224(file_path)
                print("done " + file_path)
            else:
                os.remove(file_path)
                print("remove " + file_path)


# apply resize to all
def apply_resize_to_all(dir, new_size):
    for root, dirs, files in os.walk(dir):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.splitext(file_path)[1].lower() in [".jpg", ".jpeg", ".png"] and f[0] != ".":
                resize_to_any(file_path, new_size, None)
                print("done " + file_path)
            else:
                os.remove(file_path)
                print("remove " + file_path)


# apply eq to all
def apply_eq_to_all(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.splitext(file_path)[1].lower() in [".jpg", ".jpeg", ".png"] and f[0] != ".":
                histogram_equalize(file_path)
                print("done " + file_path)
            else:
                os.remove(file_path)
                print("remove " + file_path)

# eq
def histogram_equalize(img_path):
    img = cv2.imread(img_path)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c0 = clahe.apply(img[:, :, 0])
    c1 = clahe.apply(img[:, :, 1])
    c2 = clahe.apply(img[:, :, 2])
    new_img = np.zeros(shape=img.shape)
    new_img[:, :, 0] = c0
    new_img[:, :, 1] = c1
    new_img[:, :, 2] = c2
    cv2.imwrite(img_path, new_img)


# check and remove to all
def check_and_remove_to_all(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.splitext(file_path)[1].lower() in [".jpg", ".jpeg", ".png"] and f[0] != ".":
                print("ok " + file_path)
            else:
                os.remove(file_path)
                print("remove " + file_path)


ROOT = "sundata"

if __name__ == "__main__":
    check_and_remove_to_all("sundata/all")