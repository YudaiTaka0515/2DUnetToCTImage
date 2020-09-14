import numpy as np
import cv2
import os
from Utils.LoadImage import *


def CalcCentroid(mask):
    """
    mask画像内のオブジェクトのうち、一番大きいオブジェクトの重心を返す
    :param mask: (h, w)の二次元配列。型はuint8
    :return:
    """

    if mask.dtype != np.uint8:
        print("data type is not correct")
        exit()

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    print("number of objects : ", n_labels-1)

    object_areas = []
    for i in range(1, n_labels):
        object_areas.append(stats[i][4])
        print("object area : ", stats[i][4])

    # 最大のオブジェクトのインデックスを取得(背景のインデックスが0であることを考慮する)
    max_object_index = np.argmax(object_areas) + 1
    return centroids[max_object_index]


PRED_DIR = r"I:\Data\ForTrachea\Result\SegNet_0908\predicted"
MASK_DIR = r"I:\Data\ForTrachea\MASKS"


if __name__ == '__main__':
    for i in range(5):
        patient_id = 1 + i
        # pred_path = os.path.join(PRED_DIR, "patient"+str(patient_id))
        masks_path = os.path.join(MASK_DIR, "patient"+str(patient_id))

        # pred_images = LoadImageFromPng(pred_path, patient_id, should_crop=False)
        masks = LoadImageFromPng(masks_path, patient_id, should_crop=False)

        # 0~255に変換
        # pred_images = (pred_images*255).astype(np.uint8)
        masks = (masks*255).astype(np.uint8)


        # print(pred_images.shape)
        # print(masks.shape)

        centroid = CalcCentroid(masks[0])
        print("Centroid : ", centroid)






