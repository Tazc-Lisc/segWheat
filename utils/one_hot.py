from skimage import io
from skimage.color import rgb2gray
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
import cv2

colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]

img_path='D:\\unet1\\bone\\train_new\\jian_pen\\mask_new\\'
save_img_path='D:\\unet1\\bone\\train_new\\jian_pen\\mask_new4\\'
def mask2label(img_path):
    gray =img_path
    img1 = np.expand_dims(img_path, 2)
    img2 = np.concatenate((img1, img1, img1), axis=-1)
    img2[:, :, 0] = gray
    img2[:, :, 1] = gray
    img2[:, :, 2] = gray
    width, height, c = img2.shape

    for x in range(width):
        for y in range(height):
            if img2[x, y, 0] == 0 and img2[x, y, 1] == 0 and img2[x, y, 2] == 0:
                img2[x, y, 0] = 0
                img2[x, y, 0] = 0
                img2[x, y, 0] = 0

    for x in range(width):
        for y in range(height):
            if img2[x, y, 0] == 1 and img2[x, y, 2] == 1 and img2[x, y, 2] == 1:
                img2[x, y, 0] = 255
                img2[x, y, 1] = 255
                img2[x, y, 2] = 255
    return img2
