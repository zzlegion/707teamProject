import cv2
import numpy as np
# conda install -c conda-forge opencv


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

    cv2.imwrite("normal_"+img_path, new_img)
    return new_img

if __name__ == "__main__":
    histogram_equalize("61.JPG")
