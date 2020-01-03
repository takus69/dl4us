import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import fashion_mnist, mnist


def load(dataset='f-mnist'):
    '''
    iris, mnist, fashion mnist のデータを読み込む
    デフォルトはfashion mnist
    mnist, fashion mnist は255で除算し最大値1に正規化
    iris は、MinMaxScalerで正規化
    '''
    if dataset == 'iris':
        iris = load_iris()
        x = iris.data
        y = iris.target
        mms = MinMaxScaler()
        x = mms.fit_transform(x)
    elif dataset == 'mnist':
        (x, y), (x2, y2) = mnist.load_data()
        x = np.concatenate([x, x2], axis=0)
        y = np.concatenate([y, y2], axis=0)
        x = np.expand_dims(x, axis=-1)
        x = x / 255.
    else:
        (x, y), (x2, y2) = fashion_mnist.load_data()
        x = np.concatenate([x, x2], axis=0)
        y = np.concatenate([y, y2], axis=0)
        x = np.expand_dims(x, axis=-1)
        x = x / 255.
    return x, y
