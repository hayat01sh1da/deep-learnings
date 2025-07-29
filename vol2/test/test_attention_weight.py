import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src/layers')
from attention_weight import AttentionWeight

class TestAttentionWeight(unittest.TestCase):
    def setUp(self):
        self.attention_weight = AttentionWeight()
        self.hs               = np.random.randn(10, 5, 4)
        self.h                = np.random.randn(10, 4)
        self.pycaches         = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        a = self.attention_weight.forward(self.hs, self.h)
        self.assertEqual(a.shape, (10, 5))

    def test_backward(self):
        da      = self.attention_weight.forward(self.hs, self.h)
        dhs, da = self.attention_weight.backward(da)
        self.assertEqual(dhs.shape, (10, 5, 4))
        self.assertEqual(da.shape, (10, 5))

if __name__ == '__main__':
    unittest.main()
