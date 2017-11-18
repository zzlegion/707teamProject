# encoding: utf-8
import os
import shutil
import numpy as np

# image source
POSITIVE_ROOT = "/Users/liujianwei/OneDrive - inc/CMU_2017FALL/deep_learning_10707/homework/707data/data/positive_200"
NEGATIVE_ROOT = "/Users/liujianwei/OneDrive - inc/CMU_2017FALL/deep_learning_10707/homework/707data/data/negative_200"
# train and validation root
TRAIN_ROOT = "/Users/liujianwei/OneDrive - inc/CMU_2017FALL/deep_learning_10707/homework/707data/data/train"
VAL_ROOT = "/Users/liujianwei/OneDrive - inc/CMU_2017FALL/deep_learning_10707/homework/707data/data/val"
# proportion
TRAIN_PERCENT = 0.9


if __name__ == "__main__":
    # create directory
    if os.path.exists(TRAIN_ROOT):
        shutil.rmtree(TRAIN_ROOT)
    if os.path.exists(VAL_ROOT):
        shutil.rmtree(VAL_ROOT)
    train_positive = os.path.join(TRAIN_ROOT, "positive")
    train_negative = os.path.join(TRAIN_ROOT, "negative")
    val_positive = os.path.join(VAL_ROOT, "positive")
    val_negative = os.path.join(VAL_ROOT, "negative")
    os.makedirs(train_positive)
    os.makedirs(train_negative)
    os.makedirs(val_positive)
    os.makedirs(val_negative)

    # split files and move to directory
    p_files = os.listdir(POSITIVE_ROOT)
    n_files = os.listdir(NEGATIVE_ROOT)
    np.random.shuffle(p_files)
    np.random.shuffle(n_files)

    for p_file in p_files[:int(len(p_files)*TRAIN_PERCENT)]:
        shutil.copy(os.path.join(POSITIVE_ROOT, p_file), os.path.join(train_positive, p_file))
    for p_file in p_files[int(len(p_files)*TRAIN_PERCENT):]:
        shutil.copy(os.path.join(POSITIVE_ROOT, p_file), os.path.join(val_positive, p_file))

    for n_file in n_files[:int(len(n_files) * TRAIN_PERCENT)]:
        shutil.copy(os.path.join(NEGATIVE_ROOT, n_file), os.path.join(train_negative, n_file))
    for n_file in n_files[int(len(n_files) * TRAIN_PERCENT):]:
        shutil.copy(os.path.join(NEGATIVE_ROOT, n_file), os.path.join(val_negative, n_file))
