import unittest

import numpy as np

import metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.true = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 2, 3])
        self.pred = np.array([1, 2, 3, 4, 2, 3, 4, 1, 2, 2, 3, 1])
        
    def test_acc(self):
        acc = metrics.acc(self.true, self.pred)
        self.assertEqual(acc, 7/12)
        
    def test_nmi(self):
        nmi = metrics.nmi(self.true, self.pred)
        self.assertAlmostEqual(nmi, 0.457464501)
        
    def test_ari(self):
        ari = metrics.ari(self.true, self.pred)
        self.assertAlmostEqual(ari, 0.137880987)
        
    def test_show_score(self):
        scores = metrics.show_score(self.true, self.pred)
        acc = scores['acc']
        nmi = scores['nmi']
        ari = scores['ari']
        self.assertEqual(acc, 7/12)
        self.assertAlmostEqual(nmi, 0.457464501)
        self.assertAlmostEqual(ari, 0.137880987)
        
        scores = metrics.show_score(self.true, self.pred, verbose=1)
        acc = scores['acc']
        nmi = scores['nmi']
        ari = scores['ari']
        self.assertEqual(acc, 7/12)
        self.assertAlmostEqual(nmi, 0.457464501)
        self.assertAlmostEqual(ari, 0.137880987)

