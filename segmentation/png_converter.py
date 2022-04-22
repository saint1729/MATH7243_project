import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = "/Users/sakshisuman12/Desktop/MATH7243_project/"
SEGMENTATION_ROOT_FOLDER = PROJECT_ROOT + "/segmentation/"
SEGMENTATION_PNG_ROOT_FOLDER = PROJECT_ROOT + "/segmentation_png/"


def save_png_1(jpg_file):
    # print(jpg_file.split(".")[-1])
    png_file = ".".join(jpg_file.split(".")[:-1]) + ".png"
    im = Image.open(jpg_dir_path + jpg_file)
    im.save(png_dir_path + png_file)
    # print(png_file)


def save_png_input(jpg_file):
    # print(jpg_file.split(".")[-1])
    png_file = ".".join(jpg_file.split(".")[:-1]) + ".png"
    image = cv2.imread(jpg_dir_path + jpg_file, 1)
    img_shape = image.shape
    alpha = np.full((img_shape[0], img_shape[1]), 255, dtype=np.uint8)
    image = np.dstack((image, alpha))
    cv2.imwrite(png_dir_path + png_file, image)


def save_png_output(jpg_file):
    # print(jpg_file.split(".")[-1])
    png_file = ".".join(jpg_file.split(".")[:-1]) + ".png"
    image = cv2.imread(jpg_dir_path + jpg_file, 1)
    img_shape = image.shape

    # print(np.unique(image))

    image[image < 100] = 0
    image[image >= 100] = 1

    # print(np.unique(image))
    # print(np.sum(image == 1))
    # print(np.sum(image == 0))
    # print()

    alpha = np.full((img_shape[0], img_shape[1]), 255, dtype=np.uint8)
    image = np.dstack((image, alpha))

    # print(image.shape)
    # print(np.prod(image.shape))
    # print(np.sum((image == 0)))
    # print(np.sum((image == 255)))
    # print(np.sum(image == 255))
    # print(np.unique(image).shape)
    # print()

    cv2.imwrite(png_dir_path + png_file, image)


if __name__ == "__main__":

    all_dirs = os.listdir(SEGMENTATION_ROOT_FOLDER)

    for dir_name in all_dirs:
        jpg_dir_path = SEGMENTATION_ROOT_FOLDER + dir_name + "/"
        if os.path.isdir(jpg_dir_path):
            png_dir_path = SEGMENTATION_PNG_ROOT_FOLDER + dir_name + "_png/"
            if not os.path.exists(png_dir_path):
                os.makedirs(png_dir_path)
            jpg_files = os.listdir(jpg_dir_path)
            df = pd.DataFrame(jpg_files).head(10)
            # df = pd.DataFrame(jpg_files)
            df[0].apply(save_png_output if "output" in dir_name else save_png_input)
            # break
