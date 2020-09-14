import numpy as np
import os
from Utils.LoadImage import *
from Utils.ExtractLargestObject import *


def CalcIou(pred_image, mask):
    """
    0, 1のマスク画像(一枚)であることに注意
    :param pred_image:
    :param mask:
    :return:
    """
    # TODO
    # uniqueで条件判定可能?
    unique_val = [0, 1]
    if (np.all(pred_image == 0) | np.all(pred_image == 1.0)) & (np.all(mask == 0) | np.all(mask == 1.0)):
        print("[0, 1]以外が含まれています　: CalIou")
        exit()

    tp = GetTP(pred_image, mask)
    fp = GetFP(pred_image, mask)
    fn = GetFN(pred_image, mask)
    print(tp, fp, fn)

    return tp/(tp+fp+fn)


def CalcDice(pred_image, mask):
    """
        0, 1のマスク画像(一枚)であることに注意
        :param pred_image:
        :param mask:
        :return:
        """
    # TODO
    # uniqueで条件判定可能?
    unique_val = [0, 1]
    if (np.all(pred_image == 0) | np.all(pred_image == 1.0)) & (np.all(mask == 0) | np.all(mask == 1.0)):
        print("[0, 1]以外が含まれています　: CalIou")
        exit()

    tp = GetTP(pred_image, mask)
    fp = GetFP(pred_image, mask)
    fn = GetFN(pred_image, mask)
    print(tp, fp, fn)

    return tp / (tp + (fp + fn)*0.5)


def GetTP(pred_image, mask):
    return np.count_nonzero((pred_image == 1) & (mask == 1))


def GetTN(pred_image, mask):
    return np.count_nonzero((pred_image == 0) & (mask == 0))


def GetFP(pred_image, mask):
    return np.count_nonzero((pred_image == 1) & (mask == 0))


def GetFN(pred_image, mask):
    return np.count_nonzero((pred_image == 0) & (mask == 1))


PRED_DIR = r"I:\Data\ForTrachea\Result\SegNet_0911\predicted"
MASK_DIR = r"I:\Data\ForTrachea\MASKS"

if __name__ == '__main__':
    should_remove = True

    for i in range(5):
        patient_id = 1 + i
        pred_path = os.path.join(PRED_DIR, "patient"+str(patient_id))
        masks_path = os.path.join(MASK_DIR, "patient"+str(patient_id))

        pred_images = LoadImageFromPng(pred_path, patient_id, should_crop=False)
        masks = LoadImageFromPng(masks_path, patient_id, should_crop=False)

        if should_remove:
            pred_images = (pred_images*255).astype(np.uint8)
            for j in range(pred_images.shape[0]):
                pred_images[j] = ExtractLargestObject(pred_images[j])
            pred_images = (pred_images/255).astype(float)

        iou = CalcIou(pred_images[0], masks[0])
        print("IOU : ", iou)
        dice = CalcDice(pred_images[0], masks[0])
        print("Dice: ", dice)


