from Utils.LoadImage import *
from natsort import natsorted
from keras.models import load_model
import cv2
from Utils.Metrics import dice_coefficient, dice_coefficient_loss
import matplotlib.pyplot as plt
from PIL import Image
from Utils.ExtractLargestObject import *

DIR_DST_IS_SAVED = r"I:\Data\ForTrachea\Result\SegNet_0911"
PRE_TRAINED_MODEL = os.path.join(DIR_DST_IS_SAVED, "model.h5")
PRE_TRAINED_WEIGHTS = os.path.join(DIR_DST_IS_SAVED, "weights.hdf5")

# データセット(test用)の格納場所


def Predict(patient_id):
    # とりあえずtrainingで使った1~5をPredict
    # テスト画像、マスク画像の読み込み
    test_images = LoadImageFromPng(IMAGES_FOR_TEST_DATA, patient_id, should_crop=True)
    test_masks = LoadImageFromPng(MASKS_FOR_TEST_DATA, patient_id, should_crop=True)

    n_slice = test_masks.shape[0]

    test_images = test_images.reshape((n_slice, 1, CROPPED_SHAPE["x"], CROPPED_SHAPE["y"]))
    test_masks = test_masks.reshape((n_slice, 1, CROPPED_SHAPE["x"], CROPPED_SHAPE["y"]))

    pre_trained_model = load_model(PRE_TRAINED_MODEL,
                                   custom_objects={"dice_coefficient_loss": dice_coefficient_loss,
                                                   "dice_coefficient": dice_coefficient})
    pre_trained_model.load_weights(PRE_TRAINED_WEIGHTS)

    predicted_masks = pre_trained_model.predict(test_images)
    predicted_masks = np.where(predicted_masks>0.5, 1, 0)
    test_loss = pre_trained_model.evaluate(predicted_masks, test_masks)
    print("patient {}: {}".format(patient_id, test_loss))

    for i in range(predicted_masks.shape[0]):
        predicted_masks = predicted_masks.reshape(n_slice, CROPPED_SHAPE["x"], CROPPED_SHAPE["y"], 1)
        plt.imshow(predicted_masks[i])
        plt.show()

    # 画像の保存
    if not os.path.exists(PATH_SAVED_PREDICTED):
        os.mkdir(PATH_SAVED_PREDICTED)
        print("mkdir : ", PATH_SAVED_PREDICTED)

    masks_for_save = np.zeros((n_slice, ORIGINAL_SHAPE["x"], ORIGINAL_SHAPE["y"]), dtype='uint8')
    for i in range(n_slice):
        center = (ORIGINAL_SHAPE["x"]//2, ORIGINAL_SHAPE["y"]//2)
        w = CROPPED_SHAPE["x"]
        h = CROPPED_SHAPE["y"]
        masks_for_save[i, center[0]-w//2:center[0]+w//2, center[1]-h//2:center[1]+h//2] = \
            predicted_masks[i, :, :, 0] * 255

        # 一番大きいオブジェクトのみ抽出
        # masks_for_save[i] = ExtractLargestObject(masks_for_save[i])

        file_name = str(i+1) + ".png"
        temp = np.zeros(masks_for_save[i].shape, dtype='uint8')

        data = np.stack([masks_for_save[i], temp, temp], axis=-1)

        pil_image = Image.fromarray(data)
        pil_image.save(os.path.join(PATH_SAVED_PREDICTED, file_name))


if __name__ == '__main__':
    for i in range(5):
        PATIENT_ID = i+1
        IMAGES_FOR_TEST_DATA = os.path.join(IMAGE_DIR, "patient" + str(PATIENT_ID))
        MASKS_FOR_TEST_DATA = os.path.join(MASK_DIR, "patient" + str(PATIENT_ID))
        PATH_SAVED_PREDICTED = os.path.join(DIR_DST_IS_SAVED,
                                            os.path.join("predicted", "patient"+str(i+1)))

        Predict(PATIENT_ID)
