import unittest
import sys
import os
import shutil
import glob
sys.path.append('./4_neural_network/src')
from template import Template

class TestTemplate(unittest.TestCase):
    def setUp(self) -> None:
        self.template: Template = Template()
        self.pycaches: list[str] = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self) -> None:
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

if __name__ == '__main__':
    unittest.main()
