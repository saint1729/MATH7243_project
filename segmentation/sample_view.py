import json

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

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import imageio
import shutil

import matplotlib.pyplot as plt


def convert_regions(regions):
    ans = []
    for region in eval(regions):
        ans.append((np.array([[pair['x'], pair['y']] for pair in region]) * 512).astype(int))
    return ans


if __name__ == "__main__":

    PROJECT_ROOT = "/Users/sakshisuman12/Desktop/MATH7243_project/"

    flag = False

    df = pd.read_csv(PROJECT_ROOT + "/unzipped/Hemorrhage Segmentation "
                     "Project/Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148.csv")

    images_labels = df[["Origin", "Correct Label"]]
    images_labels = images_labels[images_labels["Correct Label"].notnull()]
    images_labels["Regions"] = images_labels["Correct Label"].apply(convert_regions)

    print(images_labels.head())

    if flag:
        im = cv2.imread("ID_005f428d2_2.jpg")
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb, cmap=plt.cm.Spectral)
        plt.show()

        # canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        #
        # regions_np = []
        # for region in regions:
        #     l = []
        #     for pair in region:
        #         x, y = pair['x'], pair['y']
        #         l.append([x, y])
        #     a = np.array(l) * 512
        #     a = a.astype(int)
        #     # regions_np.append(a)
        #     # print(a[:, 0], a[:, 1])
        #     canvas[a[:, 1], a[:, 0], :] = 255

        # regions_np = np.array(regions_np)

        # print(canvas)

        # cv2.imwrite("4.jpg", canvas)
