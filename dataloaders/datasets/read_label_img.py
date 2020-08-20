import cv2
import numpy as np
from PIL import Image

filename = "iris_seg/CASIA-distance/train/SegmentationClass/S4059R18_02193.png"

cv_img = cv2.imread(filename)
print(cv_img.shape)

img = Image.open(filename)

np_img = np.array(img, dtype=np.int32)
print(np_img.shape)

label = list(set(np_img.flatten()))
print("label:", label)