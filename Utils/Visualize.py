import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2
from Utils.LoadImage import *
from Utils.Const import *
import matplotlib.pyplot as plt


def OverlayMasksOnImages(f_images, f_masks):
    if not (f_images.dtype == 'float' and f_masks.dtype == 'float'):
        print("input float numpy array")
        exit()
    if np.all(f_masks == 0) | np.all(f_masks == 1.0):
        print("[0, 1]以外が含まれています")
        exit()

    overlaid_images_ = np.zeros(f_masks.shape)
    for i in range(overlaid_images_.shape[0]):
        overlaid_images_[i] = cv2.addWeighted(f_images[i], 0.3, f_masks[i], 0.7, 0)

    return overlaid_images_


def SaveImagesAsGif(f_images, save_path):
    if not (f_images.dtype == 'float'):
        print("input float numpy array")
        exit()

    ui_images = (f_images * 255).astype('uint8')

    images_for_save = np.stack([ui_images, ui_images, ui_images], axis=-1)

    # PIL型に変換して，リストに格納
    images_list = []
    for i in range(images_for_save.shape[0]):
        images_list.append(Image.fromarray(images_for_save).convert('P'))

    # GIFを保存
    images_list[0].save(save_path,
                        save_all=True,
                        append_images=blend_images[1:],
                        optimize=False,
                        duration=100,
                        loop=5)

    print("save gif : ", save_path)


if __name__ == "__main__":
    patient_id = 1
    folder_name = "patient" + str(patient_id)
    pred_path = r"I:\Data\ForTrachea\Result\SegNet_0911\predicted"
    images_path = r"I:\Data\ForTrachea\IMAGES"
    test_images = LoadImageFromPng(os.path.join(images_path, folder_name), patient_id, should_crop=False)
    pred_masks = LoadImageFromPng(os.path.join(pred_path, folder_name), patient_id, should_crop=False)
    plt.imshow(test_images[0])
    plt.show()
    overlaid_image = test_images[0].copy()
    points = list(zip(*np.where(pred_masks[0] == 1)))
    for point in points:
        overlaid_image[point] = 1.0
    plt.imshow(overlaid_image)
    plt.show()

    data = np.stack([overlaid_image, test_images[0], test_images[0]], axis=-1)
    # plt.imshow(data)
    # plt.show()
    pil_image = Image.fromarray((data*255).astype('uint8'))
    save_path = r"I:\Data\ForTrachea\Result\SegNet_0911\blend" + str(patient_id) + ".png"
    pil_image.save(save_path)




















