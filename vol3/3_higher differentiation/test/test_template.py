import unittest
import sys
import os
import shutil
import glob
sys.path.append('./3_higher differentiation/src')
from template import Template

class TestTemplate(unittest.TestCase):
    def setUp(self):
        self.template = Template()
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

if __name__ == '__main__':
    unittest.main()
