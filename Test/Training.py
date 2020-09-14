from Model.Segnet import *
from Model.Unet2D import *
from Utils.LoadImage import *
from Utils.SetGpu import *
from Utils.Callback import *
from keras.optimizers import Adam
from Utils.Metrics import dice_coefficient_loss, dice_coefficient
from Utils.Callback import *
import tensorflow as tf

from Utils.PlotHistory import *

initial_learning_rate = 0.0001


def Training():
    # 要確認

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
        print("mkdir : ", SAVE_DIR)

    # データの読み込み

    x_train, y_train, x_validation, y_validation = LoadDataset(IMAGE_DIR, MASK_DIR, TEST_NUM)
    # 入力が正しいかチェックする
    print("-" * 20 + "check shape" + "-" * 20)
    print("train | (images, masks) = ({}, {})".format(x_train.shape, y_train.shape))
    print("validation | (images, masks) = ({}, {})".format(x_validation.shape, y_validation.shape))
    print("shape :", x_train.shape)

    # shapeを(枚数, channel, x, y)に変更
    train_shape = (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    validation_shape = (x_validation.shape[0], 1, x_validation.shape[1], x_validation.shape[2])
    x_train = x_train.reshape(train_shape)
    y_train = y_train.reshape(train_shape)
    x_validation = x_validation.reshape(validation_shape)
    y_validation = y_validation.reshape(validation_shape)

    print("reshaped : ", x_train.shape)

    # model = Unet2D(input_shape=(1, ORIGINAL_SHAPE["x"], ORIGINAL_SHAPE["y"]), n_labels=1)
    model = Unet2D(input_shape=(1, CROPPED_SHAPE["x"], CROPPED_SHAPE["y"]), n_labels=1)

    model_path = os.path.join(SAVE_DIR, "model.h5")

    print(model.summary())

    history = model.fit(x=x_train, y=y_train,
                        batch_size=N_BATCH, epochs=N_EPOCH,
                        validation_data=(x_validation, y_validation),
                        verbose=2,
                        callbacks=get_callbacks(model_file=model_path))
    model.save_weights(os.path.join(SAVE_DIR, "weights.hdf5"))

    PlotHistory(history, save_dir=SAVE_DIR)


if __name__ == '__main__':
    Training()