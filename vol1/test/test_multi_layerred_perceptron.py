import unittest
import sys
sys.path.append('./src')
from multi_layered_perceptron import MultiLayeredPerceptron

class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.mp = MultiLayeredPerceptron()

    def test_xor_gate(self):
        self.assertEqual(0, self.mp.xor_gate(0, 0))
        self.assertEqual(1, self.mp.xor_gate(1, 0))
        self.assertEqual(1, self.mp.xor_gate(0, 1))
        self.assertEqual(0, self.mp.xor_gate(1, 1))

if __name__ == '__main__':
    unittest.main()
