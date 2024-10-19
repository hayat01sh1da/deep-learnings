import unittest
import sys
import os
import shutil
import glob
sys.path.append('./src')
from simple_perceptron import SimplePerceptron

class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.sp       = SimplePerceptron()
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    def test_and_gate(self):
        self.assertEqual(0, self.sp.and_gate(0, 0))
        self.assertEqual(0, self.sp.and_gate(1, 0))
        self.assertEqual(0, self.sp.and_gate(0, 1))
        self.assertEqual(1, self.sp.and_gate(1, 1))

    def test_nand_gate(self):
        self.assertEqual(1, self.sp.nand_gate(0, 0))
        self.assertEqual(1, self.sp.nand_gate(1, 0))
        self.assertEqual(1, self.sp.nand_gate(0, 1))
        self.assertEqual(0, self.sp.nand_gate(1, 1))

    def test_or_gate(self):
        self.assertEqual(0, self.sp.or_gate(0, 0))
        self.assertEqual(1, self.sp.or_gate(1, 0))
        self.assertEqual(1, self.sp.or_gate(0, 1))
        self.assertEqual(1, self.sp.or_gate(1, 1))

if __name__ == '__main__':
    unittest.main()
