import unittest
import numpy as np
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/layers')
from time_attention import TimeAttention

class TestTimeAttention(unittest.TestCase):
    def setUp(self):
        self.time_attention = TimeAttention()
        self.hs_enc         = np.random.randn(10, 4, 4)
        self.hs_dec         = np.random.randn(10, 5, 4)
        self.pycaches       = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_forward(self):
        out = self.time_attention.forward(self.hs_enc, self.hs_dec)
        self.assertEqual(out.shape, (10, 5, 4))

    def test_backward(self):
        dout             = self.time_attention.forward(self.hs_enc, self.hs_dec)
        dhs_enc, dhs_dec = self.time_attention.backward(dout)
        self.assertEqual(dhs_enc.shape, (10, 4, 4))
        self.assertEqual(dhs_dec.shape, (10, 5, 4))

if __name__ == '__main__':
    unittest.main()
