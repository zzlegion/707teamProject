import os

TRAIN_DIR = "/Users/liujianwei/OneDrive - inc/CMU_2017FALL"

for r, d, files in os.walk(TRAIN_DIR):
    print("r", r)
    print()

    print("d", d)
    print()

    print("files", files)
    print()
