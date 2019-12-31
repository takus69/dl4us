import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def load():
    '''
    Fashion MNISTのtrainデータを読み込む
    '''
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_train = x_train / 255.
    return x_train, y_train
