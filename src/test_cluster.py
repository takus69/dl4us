import unittest

import numpy as np

from cluster import Cluster, AutoEncoder


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
        self.assertEqual(y_label.shape, (100,))
        self.assertEqual(np.min(y_label), 0)
        self.assertEqual(np.max(y_label), 9)
        
        
    def test_cluster(self):
        self.x_test = self.x_test.reshape((len(self.x_test), -1))
        self.basic_test(Cluster(n_classes=10))
        
    def test_autoencoder(self):
        self.x_test = self.x_test.reshape((len(self.x_test), -1))
        self.basic_test(AutoEncoder())

        
if __name__ == '__main__':
    unittest.main()