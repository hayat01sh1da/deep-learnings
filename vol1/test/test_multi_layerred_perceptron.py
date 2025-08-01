import unittest
import sys
import os
import shutil
import glob
sys.path.append('./src')
from multi_layered_perceptron import MultiLayeredPerceptron

class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.mp       = MultiLayeredPerceptron()
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_xor_gate(self):
        self.assertEqual(self.mp.xor_gate(0, 0), 0)
        self.assertEqual(self.mp.xor_gate(1, 0), 1)
        self.assertEqual(self.mp.xor_gate(0, 1), 1)
        self.assertEqual(self.mp.xor_gate(1, 1), 0)

if __name__ == '__main__':
    unittest.main()
