import os
import cv2
import numpy as np

root_dir = "data/imgseg/dataset-494/bak"
result_dir = "data/imgseg/dataset-494/png"

for filename in os.listdir(root_dir):
    if filename.endswith(".png"):
        print(filename)
        src_file = os.path.join(root_dir, filename)
        dst_file = os.path.join(result_dir, filename)
        src_img = cv2.imread(src_file, 0)
        dst_img = np.round(src_img).astype(np.uint8)
        print(set(dst_img.flatten()))
        cv2.imwrite(dst_file, dst_img)