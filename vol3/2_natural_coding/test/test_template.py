import unittest
import sys
import os
import shutil
import glob
sys.path.append('./2_natural_coding/src')
from template import Template

class TestTemplate(unittest.TestCase):
    def setUp(self):
        self.template = Template()
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

if __name__ == '__main__':
    unittest.main()
