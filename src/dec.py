import numpy as np
from sklearn.cluster import KMeans
from time import time
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling
import tensorflow.keras.backend as K

from cluster import Cluster
from metrics import show_score


class DEC(Cluster):
    def __init__(self, n_classes=10,
                 dims=(784, 500, 500, 2000, 10),
                 update_interval=140,
                 pretrain_epochs=300,
                ):
        super(DEC, self).__init__()

        self.n_clusters = n_classes
        self.update_interval = update_interval
        self.pretrain_epochs = pretrain_epochs
        self.autoencoder, self.encoder, self.model = self.build_model(dims)

        
    def build_model(self, dims):
        act='relu'
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
        
        n_stacks = len(dims) - 1
        # input
        x = Input(shape=(dims[0],), name='input')
        h = x

        # internal layers in encoder
        for i in range(n_stacks-1):
            h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

        # hidden layer
        h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

        y = h
        # internal layers in decoder
        for i in range(n_stacks-1, 0, -1):
            y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

        # output
        y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)
        autoencoder = Model(inputs=x, outputs=y, name='AE')
        encoder_model = Model(inputs=x, outputs=h, name='encoder')
        
        # prepare DEC model
        y_cls = ClusteringLayer(self.n_clusters, name='clustering')(h)
        model = Model(inputs=x, outputs=y_cls)

        return autoencoder, encoder_model, model
    
    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, verbose=0):
        if verbose > 0:
            print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=verbose)
        if verbose > 0:
            print('Pretraining time: %ds' % round(time() - t0))

    def target_distribution(self, q):
        '''
        モデルから算出された各クラスタの確率(q)をクラスタごとの偏りを平準化する
        '''
        weight = q ** 2 / q.sum(axis=0)
        return (weight.T / weight.sum(axis=1)).T

    def train(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, verbose=0):
        # Step 1: initialize cluster centers using k-means
        t1 = time()
        if verbose > 0:
            print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None and verbose > 0:
                    scores = show_score(y, y_pred)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (
                        ite, scores['acc'], scores['nmi'], scores['ari']), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    if verbose > 0:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        return y_pred
    
    def _fit(self, x, y, verbose):
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
        self.pretrain(x=x, y=y, optimizer=pretrain_optimizer,
                      epochs=self.pretrain_epochs, verbose=verbose)
        self.model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
        self.train(x, y=y, update_interval=self.update_interval, verbose=verbose)

    def _predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x)
        return q.argmax(axis=1)


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.initial_weights = weights

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.clusters = self.add_weight(shape=(self.n_clusters, int(input_dim)), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        '''
        t分布により一番近いクラスタを判断する
        ベクトル - クラスタ中心がt分布に従うと仮定
        '''
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)))
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

