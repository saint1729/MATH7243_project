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

PROJECT_ROOT = "/Users/sakshisuman12/Desktop/MATH7243_project/"
RENDERS_FOLDER = PROJECT_ROOT + "/unzipped/XN_project/renders/"
SEGMENTATION_LABEL_FOLDER = PROJECT_ROOT + "/unzipped/Hemorrhage Segmentation Project/"
SEGMENTATION_INPUT_FOLDER = PROJECT_ROOT + "/segmentation/input/"
SEGMENTATION_OUTPUT_FOLDER = PROJECT_ROOT + "/segmentation/output/"

SEGMENTATION_VISUALIZE_FOLDER = PROJECT_ROOT + "/segmentation/visualize/"

SHADES = ["brain_bone_window", "brain_window", "max_contrast_window", "subdural_window"]
SEGMENTATION_CSV_FILES = ["Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148.csv",
                          "Results_Intraparenchymal Hemorrhage Detection_2020-11-16_21.39.31.268.csv",
                          "Results_Multiple Hemorrhage Detection_2020-11-16_21.36.24.018.csv",
                          "Results_Subarachnoid Hemorrhage Detection_2020-11-16_21.36.18.668.csv",
                          "Results_Subdural Hemorrhage Detection_2020-11-16_21.35.48.040.csv",
                          "Results_Subdural Hemorrhage Detection_2020-11-16_21.37.19.745.csv"]

DIRECTORY_STRUCTURE = {}


def convert_regions(regions):
    ans = []
    for region in eval(regions):
        ans.append((np.array([[pair['x'], pair['y']] for pair in region]) * 512).astype(int))
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
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
                shade_path = class_path + "/" + shade
                if not os.path.exists(shade_path):
                    os.makedirs(shade_path)
                os.system(f"cp {RENDERS_FOLDER}/{class_name}/{shade}/{image_name} {shade_path}/{image_name}")

    # all_classes = "__".join(list(class_set))

    # for class_name in class_set:
    #     for shade in shade_dict[class_name]:
    #         output_image_name = output_image_name + "__" + class_name + "__" + shade

    if len(class_set) != 1:
        print(image_name)


def assign_class_names(image_name):
    class_set = set()

    for class_name, shades in DIRECTORY_STRUCTURE.items():
        for shade, shade_files in shades.items():
            if image_name in shade_files:
                class_set.add(class_name)

    return list(class_set)


def create_output_canvas(filename, regions, class_names):
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)

    for region in regions:
        if region.shape != (0,):
            canvas[region[:, 1], region[:, 0], :] = 255

    if not os.path.exists(SEGMENTATION_OUTPUT_FOLDER):
        os.makedirs(SEGMENTATION_OUTPUT_FOLDER)

    cv2.imwrite(SEGMENTATION_OUTPUT_FOLDER + "/" + filename, canvas)

    for class_name in class_names:
        if not os.path.exists(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name):
            os.makedirs(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name)
        for shade in SHADES:
            if not os.path.exists(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name + "/" + shade):
                os.makedirs(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name + "/" + shade)
            image = cv2.imread(SEGMENTATION_INPUT_FOLDER + "/" + class_name + "/" + shade + "/" + filename)
            for region in regions:
                if region.shape != (0,):
                    image[region[:, 1], region[:, 0], 2] = 255
            cv2.imwrite(SEGMENTATION_VISUALIZE_FOLDER + "/" + class_name + "/" + shade + "/" + filename, image)


if __name__ == "__main__":
    # start_time = time.time()
    populate_paths()
    # print(time.time() - start_time)

    # start_time = time.time()
    # for i in range(1000000):
    #     create_image("ID_004c4b319.jpg")
    # print(time.time() - start_time)

    for csv_file_name in SEGMENTATION_CSV_FILES:
        df = pd.read_csv(SEGMENTATION_LABEL_FOLDER + "/" + csv_file_name)
        images_labels = df[["Origin", "Correct Label"]]
        images_labels = images_labels[images_labels["Correct Label"].notnull()]
        images_labels["Regions"] = images_labels["Correct Label"].apply(convert_regions)
        images_labels["Classes"] = images_labels["Origin"].apply(assign_class_names)
        images_labels.apply(lambda row: create_output_canvas(row["Origin"], row["Regions"], row["Classes"]), axis=1)
        # print(images_labels.head())
