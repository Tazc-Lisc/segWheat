import argparse
import json
import os
import time
import cv2
from PIL import Image
from tqdm import tqdm
from datetime import date
import numpy as np
import matplotlib as plt
import sys
import glob


class PolygonMaskConversion:
    def __init__(self, epsilon_factor=0.001):
        self.epsilon_factor = epsilon_factor

    def reset(self):
        self.custom_data = dict(
            version=VERSION,
            flags={},
            shapes=[],
            imagePath="",
            imageData=None,
            imageHeight=-1,
            imageWidth=-1,
        )

    def get_image_size(self, image_file):
        with Image.open(image_file) as img:
            width, height = img.size
            return width, height
    def polygon_to_mask(self, img_file, mask_file, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        polygons = []
        for shape in data["shapes"]:
            points = shape["points"]
            polygon = []
            for point in points:
                x, y = point
                polygon.append((x, y))
            polygons.append(polygon)

        image_width, image_height = self.get_image_size(img_file)
        image_shape = (image_height, image_width)
        binary_mask = np.zeros(image_shape, dtype=np.uint8)
        for polygon_points in polygons:
            np_polygon = np.array(polygon_points, np.int32)
            np_polygon = np_polygon.reshape((-1, 1, 2))
            cv2.fillPoly(binary_mask, [np_polygon], color=255)
        cv2.imwrite(mask_file, binary_mask)


if __name__ == "__main__":
    converter = PolygonMaskConversion(epsilon_factor=0.001)
    os.makedirs('./data/', exist_ok=True)
    file_list = glob.glob('./data/*.JPG')
    for file_name in tqdm( file_list):
        img_file = file_name
        json_file = file_name.replace('D.JPG', 'MS_RE.json')
        mask_file = file_name.replace('D.JPG', 'MS_RE.png')
        converter.polygon_to_mask(img_file, mask_file, json_file)
    print('over')
