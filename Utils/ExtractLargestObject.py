import numpy as np
import cv2
import os
from Utils.LoadImage import *


def ExtractLargestObject(mask):
    """
    mask画像内のオブジェクトのうち、一番大きいオブジェクトのみを抽出する
    :param mask: (h, w)の二次元配列。型はuint8
    :return:
    """

    if mask.dtype != np.uint8:
        print("data type is not correct")
        exit()

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    object_areas = []
    for i in range(1, n_labels):
        object_areas.append(stats[i][4])

    # 最大のオブジェクトのインデックスを取得(背景のインデックスが0であることを考慮する)
    max_object_index = np.argmax(object_areas) + 1

    extracted_object = np.where(labels == max_object_index, 255, 0)

    return extracted_object

