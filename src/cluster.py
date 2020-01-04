import numpy as np
from sklearn.cluster import KMeans
from time import time
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from metrics import show_score


class Cluster:
    '''
    クラスタリングのクラス
    k-meansのクラスタリング手法を提供
    '''
    def __init__(self, n_classes=10):
        self.n_classes = 10
        self.kmeans = KMeans(n_clusters=n_classes)

    def _fit(self, x, y, verbose):
        self.kmeans.fit(x)
        
    def fit(self, x, y=None, verbose=0):
        '''
        x: クラスタリング対象のデータ
        y: クラスタリングの正解データ
        ver
        '''
        s_time = time()
        self._fit(x, y, verbose)
        scores = None
        if y is not None:
            scores = show_score(self._predict(x), y)
        if verbose > 0:
            print('Elapsed time: {:.3f}s, scores: acc: {:.3f}, nmi: {:.3f}, ari: {:.3f}'.format(
                time() - s_time, scores['acc'], scores['nmi'], scores['ari']
            ))
        return scores

    def _predict(self, x):
        return self.kmeans.predict(x)
    
    def predict(self, x):
        return self._predict(x)

class AutoEncoder(Cluster):
    def __init__(self, n_classes=10, dims=(784, 500, 500, 2000, 10)):
        self.n_classes = 10
        self.dims = dims
        self.kmeans = KMeans(n_clusters=n_classes)
        self.model, self.encoder_model = self.build_model()
        
    def build_model(self):
        img_dim = self.dims[0]
        hid_dim = self.dims[-1]

        # encoder
        encoder_input = Input(shape=(img_dim,))
        encoder_x = encoder_input
        for dim in self.dims[1:-1]:
            encoder_x = Dense(dim, activation='relu')(encoder_x)
        encoder_x = Dense(hid_dim)(encoder_x)
        encoder_model = Model(encoder_input, encoder_x)

        # decoder
        decoder_input = Input(shape=(hid_dim,))
        decoder_x = decoder_input
        for dim in reversed(self.dims[1:-1]):
            decoder_x = Dense(dim, activation='relu')(decoder_x)
        decoder_x = Dense(img_dim, activation='sigmoid')(decoder_x)
        decoder_model = Model(decoder_input, decoder_x)

        model = Model(encoder_input, decoder_model(encoder_x))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model, encoder_model
    
    def _fit(self, x, y, verbose):
        self.model.fit(x, x, epochs=50, batch_size=128, validation_split=0.1, verbose=verbose)
        
    def _predict(self, x):
        x_vec = self.encoder_model.predict(x)
        self.kmeans.fit(x_vec)
        y_pred = self.kmeans.predict(x_vec)
        return y_pred
