import numpy as np
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.initializers import VarianceScaling, RandomUniform
from tensorflow.keras.optimizers import SGD
import math
from sklearn import mixture
from sklearn.cluster import KMeans
from time import time

from cluster import Cluster
import metrics

import warnings
warnings.filterwarnings("ignore")

class VaDE(Cluster):
    def __init__(self, n_clusters=10,
                 input_dim = 784,
                 hidden_dim = [500, 500, 2000],
                 latent_dim = 10,
                 init=None,
                 act='relu',
                 epochs=10,
                 pretrain_epochs=10,
                 batch_size=100,
                 lr=0.002
                ):
        super(VaDE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.pretrain_epochs = pretrain_epochs
        self.epochs = epochs
        self.lr = lr

        self.n_clusters = n_clusters
        self.theta_p=self.floatX(np.ones(self.n_clusters)/self.n_clusters)  # クラスタの事前確率
        self.u_p=self.floatX(np.zeros((self.latent_dim,self.n_clusters)))  # クラスタのベクトル成分ごとの平均
        self.lambda_p=self.floatX(np.ones((self.latent_dim,self.n_clusters)))  # クラスタのベクトル成分ごとの分散
        if init is None:
            init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
        self.autoencoder, self.model, self.encoder_model = self.build_model(act=act, init=init)
            
        adam_nn= Adam(lr=self.lr,epsilon=1e-4)
        self.model.compile(optimizer=adam_nn, loss=None)
    
    def floatX(self, X):
        return K.variable(X, dtype='float32')

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def build_model(self, act, init):
        # autoencoder
        x = Input(shape=(self.input_dim,))
        h = x
        for dim in self.hidden_dim:
            h = Dense(dim, activation=act, kernel_initializer=init)(h)
        h = Dense(self.latent_dim, kernel_initializer=init)(h)
        for dim in reversed(self.hidden_dim):
            h = Dense(dim, activation=act, kernel_initializer=init)(h)
        y = Dense(self.input_dim, activation='sigmoid', kernel_initializer=init)(h)
        autoencoder = Model(x, y)
        
        # model
        x = Input(batch_shape=(self.batch_size, self.input_dim))
        h = x
        for dim in self.hidden_dim:
            h = Dense(dim, activation='relu', kernel_initializer=init)(h)
        z_mean = Dense(self.latent_dim, kernel_initializer=init)(h)
        z_log_var = Dense(self.latent_dim, kernel_initializer=init)(h)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        h_decoded = z
        for dim in reversed(self.hidden_dim):
            h_decoded = Dense(dim, activation='relu', kernel_initializer=init)(h_decoded)
        x_decoded = Dense(self.input_dim, activation='sigmoid', kernel_initializer=init)(h_decoded)
        y = ElboLayer(
            n_clusters=self.n_clusters, batch_size=self.batch_size,
            input_dim=self.input_dim, latent_dim=self.latent_dim
        )([x, x_decoded, z, z_mean, z_log_var])
        model = Model(x, y)
        encoder_model = Model(x, z_mean)
        
        return autoencoder, model, encoder_model

    def pretrain(self, x, verbose=0):
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
        #self.autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.autoencoder.fit(x, x, batch_size=self.batch_size, epochs=self.pretrain_epochs, verbose=verbose)
        self.model.layers[1].set_weights(self.autoencoder.layers[1].get_weights())
        self.model.layers[2].set_weights(self.autoencoder.layers[2].get_weights())
        self.model.layers[3].set_weights(self.autoencoder.layers[3].get_weights())
        self.model.layers[4].set_weights(self.autoencoder.layers[4].get_weights())
        self.model.layers[-2].set_weights(self.autoencoder.layers[-1].get_weights())
        self.model.layers[-3].set_weights(self.autoencoder.layers[-2].get_weights())
        self.model.layers[-4].set_weights(self.autoencoder.layers[-3].get_weights())
        self.model.layers[-5].set_weights(self.autoencoder.layers[-4].get_weights())
        sample = self.encoder_model.predict(x,batch_size=self.batch_size)
        g = mixture.GaussianMixture(n_components=self.n_clusters,covariance_type='diag')
        g.fit(sample)
        u_p = self.floatX(g.means_.T)
        lambda_p = self.floatX(g.covariances_.T)
        self.model.layers[-1].set_weights([K.eval(self.theta_p),K.eval(u_p),K.eval(lambda_p)])
        
    def lr_decay(self, epoch):
        if epoch % 10 == 0 and epoch > 0:
            self.lr *= 0.9
        return max(self.lr, 0.0002)
    
    class MonitorScore(Callback):
        def set_config(self, x, y=None, batch_size=None, verbose=0):
            self.verbose = verbose
            self.x = x
            self.y = y
            self.batch_size = batch_size
            
        def on_epoch_begin(self, epoch, logs={}):
            self.s_time = time()
            if self.verbose > 0 and self.y is not None:
                y_pred = self.model.predict(self.x, batch_size=self.batch_size)
                y_pred = np.argmax(y_pred, axis=1)
                scores= metrics.show_score(self.y, y_pred)
                print('Elapsed time: {:.3f}s, scores: acc: {:.3f}, nmi: {:.3f}, ari: {:.3f}'.format(
                    time() - self.s_time, scores['acc'], scores['nmi'], scores['ari']
                ))

    def _fit(self, x, y, verbose):
        self.pretrain(x, verbose=verbose)
        lr_scheduler = LearningRateScheduler(self.lr_decay, verbose=verbose)
        monitor_score = VaDE.MonitorScore()
        monitor_score.set_config(x=x, y=y, batch_size=self.batch_size, verbose=verbose)
        self.model.fit(x, shuffle=True, epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[lr_scheduler, monitor_score], verbose=verbose)
        
    def _predict(self, x): 
        y_pred = self.model.predict(x, batch_size=self.batch_size)
        return np.argmax(y_pred, axis=1)


class ElboLayer(Layer):
    '''
    3つの学習可能な変数を追加: theta_p,u_p,lambda_p
    '''
    def __init__(self, n_clusters, batch_size, input_dim, latent_dim, **kwargs):
        self.is_placeholder = True
        super(ElboLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def build(self, input_shape):
        assert len(input_shape) == 5
        initializer = RandomUniform(minval=0.0, maxval=1.0)
        self.theta_p = self.add_weight(shape=(self.n_clusters,), initializer=initializer, name='theta_p')
        self.u_p = self.add_weight(shape=(self.latent_dim, self.n_clusters), initializer=initializer, name='u_p')
        self.lambda_p = self.add_weight(shape=(self.latent_dim, self.n_clusters), initializer=initializer, name='lambda_p')
        self.built = True
        
    def vae_loss(self, x, x_decoded_mean, z, z_mean, z_log_var):
        gamma = self.compute_p_c_z(z)
        gamma_t=K.repeat(gamma,self.latent_dim)
        
        u_tensor3 = self.compute_u_tensor3()
        lambda_tensor3 = self.compute_lambda_tensor3()
        
        assert z_mean.shape[1:] == (self.latent_dim,), 'z_mean.shape[1:] {} != {}'.format(z_mean.shape[1:], (self.latent_dim,))
        z_mean_t=K.permute_dimensions(K.repeat(z_mean,self.n_clusters),[0,2,1])
        assert z_mean_t.shape[1:] == (self.latent_dim, self.n_clusters), 'z_mean_t.shape[1:] {} != {}'.format(z_mean_t.shape[1:], (self.latent_dim, self.n_clusters))

        assert z_log_var.shape[1:] == (self.latent_dim,), 'z_log_var.shape[1:] {} != {}'.format(z_log_var.shape[1:], (self.latent_dim,))
        z_log_var_t=K.permute_dimensions(K.repeat(z_log_var,self.n_clusters),[0,2,1])
        assert z_log_var_t.shape[1:] == (self.latent_dim, self.n_clusters), 'z_log_var_t.shape[1:] {} != {}'.format(z_log_var_t.shape[1:], (self.latent_dim, self.n_clusters))

        loss=self.input_dim * losses.binary_crossentropy(x, x_decoded_mean)\
        +K.sum(0.5*gamma_t*(self.latent_dim*K.log(math.pi*2)+K.log(lambda_tensor3)+K.exp(z_log_var_t)/lambda_tensor3+K.square(z_mean_t-u_tensor3)/lambda_tensor3),axis=(1,2))\
        -0.5*K.sum(z_log_var+1,axis=-1)\
        -K.sum(K.log(K.repeat_elements(K.expand_dims(self.theta_p, axis=0),self.batch_size,0))*gamma,axis=-1)\
        +K.sum(K.log(gamma)*gamma,axis=-1)

        return loss

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        z = inputs[2]
        z_mean = inputs[3]
        z_log_var = inputs[4]
        loss = self.vae_loss(x, x_decoded, z, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs) # Layer class のadd_lossを利用
        
        return self.compute_p_c_z(z)
    
    def compute_p_c_z(self, z):
        assert z.shape[1:] == (self.latent_dim,), 'z.shape[1:] {} != {}'.format(z.shape[1:], (self.latent_dim,))
        Z=K.permute_dimensions(K.repeat(z,self.n_clusters),[0,2,1])
        assert Z.shape[1:] == (self.latent_dim, self.n_clusters), 'Z.shape[1:] {} != {}'.format(Z.shape[1:], (self.latent_dim, self.n_clusters))

        u_tensor3 = self.compute_u_tensor3()
        lambda_tensor3 = self.compute_lambda_tensor3()

        assert self.theta_p.shape == (self.n_clusters,), 'self.theta_p.shape {} != {}'.format(self.theta_p.shape, (self.n_clusters,))
        theta_tensor3=K.expand_dims(K.expand_dims(self.theta_p, axis=0), axis=0)*K.ones((self.batch_size,self.latent_dim,self.n_clusters))
        assert theta_tensor3.shape == (self.batch_size, self.latent_dim, self.n_clusters), 'theta_tensor3.shape {} != {}'.format(
            theta_tensor3.shape, (self.batch_size, self.latent_dim, self.n_clusters))

        p_c_z=K.exp(K.sum((K.log(theta_tensor3)-0.5*K.log(2*math.pi*lambda_tensor3)-\
                           K.square(Z-u_tensor3)/(2*lambda_tensor3)),axis=1))+1e-10
        assert p_c_z.shape[1:] == (self.n_clusters,), 'p_c_z.shape[1:] {} != {}'.format(p_c_z.shape[1:], (self.n_clusters,))
        return p_c_z/K.sum(p_c_z,axis=-1,keepdims=True)
    
    def compute_u_tensor3(self):
        assert self.u_p.shape == (self.latent_dim, self.n_clusters), 'self.u_p.shape {} != {}'.format(self.u_p.shape, (self.latent_dim, self.n_clusters))
        u_tensor3=K.permute_dimensions(K.repeat(self.u_p, self.batch_size), [1, 0, 2])
        assert u_tensor3.shape == (self.batch_size, self.latent_dim, self.n_clusters), 'u_tensor3.shape {} != {}'.format(
            u_tensor3.shape, (self.batch_size, self.latent_dim, self.n_clusters))
        return u_tensor3
    
    def compute_lambda_tensor3(self):
        assert self.lambda_p.shape == (self.latent_dim, self.n_clusters), 'self.lambda_p.shape {} != {}'.format(self.lambda_p.shape, (self.latent_dim, self.n_clusters))
        lambda_tensor3=K.permute_dimensions(K.repeat(self.lambda_p, self.batch_size), [1, 0, 2])
        assert lambda_tensor3.shape == (self.batch_size, self.latent_dim, self.n_clusters), 'lambda_tensor3.shape {} != {}'.format(
            lambda_tensor3.shape, (self.batch_size, self.latent_dim, self.n_clusters))
        return lambda_tensor3