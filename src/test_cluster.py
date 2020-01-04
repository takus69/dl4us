import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from cluster import Cluster, AutoEncoder
from dec import DEC
from vade import VaDE


class TestCluster(unittest.TestCase):
    def setUp(self):
        '''
        テストデータとして(100, 10)のテストデータを作成
        '''
        self.x_test = [np.expand_dims(np.random.rand(28, 28), axis=-1) for _ in range(100)]
        self.x_test = np.array(self.x_test)
        self.y_test = np.random.randint(0, 10, 100)
        self.assertEqual(self.x_test.shape, (100, 28, 28, 1))
        self.assertEqual(self.y_test.shape, (100,))

        # irisデータ
        iris = load_iris()
        self.x_test = iris.data
        self.y_test = iris.target
        self.assertEqual(self.x_test.shape, (150, 4))
        self.assertEqual(self.y_test.shape, (150,))
        mms = MinMaxScaler()
        self.x_test = mms.fit_transform(self.x_test)
        
    def basic_test(self, cluster):
        # fit
        scores = cluster.fit(self.x_test, self.y_test)
        acc = scores['acc']
        nmi = scores['nmi']
        ari = scores['ari']
        self.assertTrue(acc > 0 and acc < 1)
        self.assertTrue(nmi > 0 and nmi < 1)
        self.assertTrue(ari > -1 and ari < 1)
        
        scores = cluster.fit(self.x_test, self.y_test, verbose=1)
        
        # predict
        y_label = cluster.predict(self.x_test)
        self.assertEqual(y_label.shape, (150,))
        self.assertEqual(np.min(y_label), 0)
        self.assertEqual(np.max(y_label), 2)
        
        
    def test_cluster(self):
        # self.x_test = self.x_test.reshape((len(self.x_test), -1))
        print('k-means')
        self.basic_test(Cluster(n_classes=3))
        
    def test_autoencoder(self):
        print('AutoEncoder')
        # self.x_test = self.x_test.reshape((len(self.x_test), -1))
        self.basic_test(AutoEncoder(n_classes=3, dims=(4, 500, 500, 2000, 10)))

    def test_dec(self):
        print('DEC')
        # self.x_test = self.x_test.reshape((len(self.x_test), -1))
        self.basic_test(DEC(n_classes=3, dims=(4, 500, 500, 2000, 10), pretrain_epochs=10))
    
    def test_vade(self):
        print('VaDE')
        # self.x_test = self.x_test.reshape((len(self.x_test), -1))
        self.basic_test(VaDE(
            n_clusters=3, hidden_dim=[100, 100, 500], input_dim=4, latent_dim=3,
            batch_size=30, init='glorot_uniform'))

        
if __name__ == '__main__':
    unittest.main()