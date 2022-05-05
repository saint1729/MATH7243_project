import json
import time

import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

from test_utils import summary, comparator

import os

import pandas as pd

import imageio
import shutil

import matplotlib.pyplot as plt

FACTOR = 1
SIZE = 512 // FACTOR

PROJECT_ROOT = "/Users/sakshisuman12/Desktop/MATH7243_project/"
RENDERS_FOLDER = PROJECT_ROOT + "/unzipped/XN_project/renders/"
SEGMENTATION_LABEL_FOLDER = PROJECT_ROOT + "/unzipped/Hemorrhage Segmentation Project/"

SEGMENTATION_INPUT_FOLDER = PROJECT_ROOT + "/segmentation/input/"
SEGMENTATION_INPUT_PREFIX = PROJECT_ROOT + "/segmentation/input"
SEGMENTATION_OUTPUT_FOLDER = PROJECT_ROOT + "/segmentation/output_" + str(FACTOR) + "/"
SEGMENTATION_VISUALIZE_FOLDER = PROJECT_ROOT + "/segmentation/visualize_" + str(FACTOR) + "/"
SEGMENTATION_VISUALIZE_PREFIX = PROJECT_ROOT + "/segmentation/visualize_" + str(FACTOR)

SHADES = ["brain_bone_window", "brain_window", "max_contrast_window", "subdural_window"]

SEGMENTATION_CSV_FILES = ["Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148.csv",
                          "Results_Intraparenchymal Hemorrhage Detection_2020-11-16_21.39.31.268.csv",
                          "Results_Multiple Hemorrhage Detection_2020-11-16_21.36.24.018.csv",
                          "Results_Subarachnoid Hemorrhage Detection_2020-11-16_21.36.18.668.csv",
                          "Results_Subdural Hemorrhage Detection_2020-11-16_21.35.48.040.csv",
                          "Results_Subdural Hemorrhage Detection_2020-11-16_21.37.19.745.csv"]

TYPE = 2

# SEGMENTATION_CSV_FILES = ["Results_Subdural Hemorrhage Detection_2020-11-16_21.35.48.040.csv",
#                           "Results_Subdural Hemorrhage Detection_2020-11-16_21.37.19.745.csv"]

DIRECTORY_STRUCTURE = {}


def convert_regions(regions):
    ans = []
    for region in eval(regions):
        ans.append((np.array([[pair['x'], pair['y']] for pair in region]) * SIZE).astype(int))
    return ans


def populate_paths():
    classes = os.listdir(RENDERS_FOLDER)

    for class_name in classes:
        folder_path = RENDERS_FOLDER + class_name
        if os.path.isdir(folder_path):
            shades = {}
            for shade in SHADES:
                shade_path = folder_path + "/" + shade
                if not os.path.exists(shade_path):
                    print(f"{shade_path} doesn't exist!")
                else:
                    shades[shade] = set(os.listdir(shade_path))
            DIRECTORY_STRUCTURE[class_name] = shades

    return DIRECTORY_STRUCTURE


def create_image(image_name):
    output_image_name = ".".join(image_name.split(".")[:-1])

    class_set = set()
    shade_dict = {}

    output_names = []

    for class_name, shades in DIRECTORY_STRUCTURE.items():
        shade_dict[class_name] = set()
        class_path = SEGMENTATION_INPUT_FOLDER + "/" + class_name
        for shade, shade_files in shades.items():
            if image_name in shade_files:
                class_set.add(class_name)
                # shade_dict[class_name].add(shade)
                # output_names.append(output_image_name + "__" + class_name + "__" + shade + ".jpg")
                shade_path = class_path + "/" + shade
                if TYPE == 0:
                    if not os.path.exists(class_path):
                        os.makedirs(class_path)
                    if not os.path.exists(shade_path):
                        os.makedirs(shade_path)
                    os.system(f"cp {RENDERS_FOLDER}/{class_name}/{shade}/{image_name} {shade_path}/{image_name}")
                elif TYPE == 1:
                    os.system(f"cp {RENDERS_FOLDER}/{class_name}/{shade}/{image_name} {SEGMENTATION_INPUT_FOLDER}/"
                              f"{output_image_name}_{shade}.jpg")
                elif TYPE == 2:
                    if not os.path.exists(SEGMENTATION_INPUT_PREFIX + "_" + shade):
                        os.makedirs(SEGMENTATION_INPUT_PREFIX + "_" + shade)
                    os.system(f"cp {RENDERS_FOLDER}/{class_name}/{shade}/{image_name} {SEGMENTATION_INPUT_PREFIX}"
                              f"_{shade}/{image_name}")

    # all_classes = "__".join(list(class_set))

    # for class_name in class_set:
    #     for shade in shade_dict[class_name]:
    #         output_image_name = output_image_name + "__" + class_name + "__" + shade

    if len(class_set) != 1:
        print(list(class_set), image_name)


def assign_class_names(image_name):
    class_set = set()

    for class_name, shades in DIRECTORY_STRUCTURE.items():
        for shade, shade_files in shades.items():
            if image_name in shade_files:
                class_set.add(class_name)

    return list(class_set)


def create_output_canvas(filename, regions, class_names):
    try:
        output_file_name = ".".join(filename.split(".")[:-1])
        canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

        for region in regions:
            if region.shape != (0,):
                # canvas[region[:, 1], region[:, 0], :] = 255
                cv2.fillPoly(canvas, pts=[region], color=(255, 255, 255))

        cv2.imwrite(SEGMENTATION_OUTPUT_FOLDER + "/" + filename, canvas)

        for class_name in class_names:
            if TYPE == 0:
                if not os.path.exists(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name):
                    os.makedirs(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name)
            for shade in SHADES:
                image = np.array([])

                if TYPE == 0:
                    if not os.path.exists(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name + "/" + shade):
                        os.makedirs(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name + "/" + shade)
                    image = cv2.imread(SEGMENTATION_INPUT_FOLDER + "/" + class_name + "/" + shade + "/" + filename)
                elif TYPE == 1:
                    image = cv2.imread(SEGMENTATION_INPUT_FOLDER + "/" + output_file_name + "_" + shade + ".jpg")
                elif TYPE == 2:
                    if not os.path.exists(SEGMENTATION_VISUALIZE_PREFIX + "_" + shade):
                        os.makedirs(SEGMENTATION_VISUALIZE_PREFIX + "_" + shade)
                    image = cv2.imread(SEGMENTATION_INPUT_PREFIX + "_" + shade + "/" + filename)

                scale_percent = 100 / FACTOR

                # calculate the 50 percent of original dimensions
                width = int(image.shape[1] * scale_percent / 100)
                height = int(image.shape[0] * scale_percent / 100)

                output_size = (width, height)
                output_img = cv2.resize(image, output_size)

                for region in regions:
                    if region.shape != (0,):
                        output_img[region[:, 1], region[:, 0], 0] = 0
                        output_img[region[:, 1], region[:, 0], 1] = 0
                        output_img[region[:, 1], region[:, 0], 2] = 255
                        # cv2.fillPoly(output_img, pts=[region], color=(0, 0, 255))

                if TYPE == 0:
                    cv2.imwrite(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name + "/" + shade + "/" + filename,
                                output_img)
                elif TYPE == 1:
                    cv2.imwrite(SEGMENTATION_VISUALIZE_FOLDER + "/" + output_file_name + "_" + shade + ".jpg",
                                output_img)
                elif TYPE == 2:
                    cv2.imwrite(SEGMENTATION_VISUALIZE_PREFIX + "_" + shade + "/" + filename, output_img)
    except:
        print(filename)


if __name__ == "__main__":
    # start_time = time.time()
    populate_paths()
    # print(time.time() - start_time)

    # start_time = time.time()
    # for i in range(1000000):
    #     create_image("ID_004c4b319.jpg")
    # print(time.time() - start_time)

    if TYPE == 1:
        if not os.path.exists(SEGMENTATION_INPUT_FOLDER):
            os.makedirs(SEGMENTATION_INPUT_FOLDER)

        if not os.path.exists(SEGMENTATION_VISUALIZE_FOLDER):
            os.makedirs(SEGMENTATION_VISUALIZE_FOLDER)

    if not os.path.exists(SEGMENTATION_OUTPUT_FOLDER):
        os.makedirs(SEGMENTATION_OUTPUT_FOLDER)

    for csv_file_name in SEGMENTATION_CSV_FILES:
        df = pd.read_csv(SEGMENTATION_LABEL_FOLDER + "/" + csv_file_name)
        images_labels = df[["Origin", "Correct Label"]]
        images_labels = images_labels[images_labels["Correct Label"].notnull()]
        images_labels["Regions"] = images_labels["Correct Label"].apply(convert_regions)
        images_labels["Origin"].apply(create_image)
        images_labels["Classes"] = images_labels["Origin"].apply(assign_class_names)
        images_labels.apply(lambda row: create_output_canvas(row["Origin"], row["Regions"], row["Classes"]), axis=1)
        # break
        # print(images_labels.columns)
        # print(images_labels["Regions"].head())
        # print(images_labels.head(5))
