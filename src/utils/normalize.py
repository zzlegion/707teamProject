import cv2
# conda install -c conda-forge opencv


def histogram_equalize(img):
    img = cv2.imread('tsukuba_l.png',0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    cv2.imwrite('clahe_2.jpg', cl1)
    return cl1