import unittest

import numpy as np

import dataset

class TestDataset(unittest.TestCase):
    def test_load(self):
        x, y = dataset.load()
        self.assertEqual(x.shape, (60000, 28, 28, 1))
        self.assertEqual(y.shape, (60000,))
        self.assertEqual(np.min(x), 0)
        self.assertEqual(np.max(x), 1)
        self.assertEqual(np.min(y), 0)
        self.assertEqual(np.max(y), 9)


if __name__ == '__main__':
    unittest.main()
