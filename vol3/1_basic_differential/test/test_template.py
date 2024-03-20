import unittest
import sys
sys.path.append('./1_basic_differential/src')
from template import Template
import numpy as np
from numpy.testing import assert_array_equal

class TestTemplate(unittest.TestCase):
    def setUp(self):
        self.template = Template()

if __name__ == '__main__':
    unittest.main()
