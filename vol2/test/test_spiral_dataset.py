import unittest
from os import path
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/concerns')
from spiral_dataset import SpiralDataset

class TestSpiralDataset(unittest.TestCase):
    def setUp(self):
        self.spiral_dataset = SpiralDataset()
        self.pycaches       = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_x_shape(self):
        self.assertEqual(self.spiral_dataset.x.shape, (300, 2))

    def test_t_shape(self):
        self.assertEqual(self.spiral_dataset.t.shape, (300, 3))

    def test_save_plot_image(self):
        file_path = '../img/spiral_plot.png'
        self.spiral_dataset.save_plot_image(file_path)
        self.assertTrue(os.path.exists(file_path))

if __name__ == '__main__':
    unittest.main()
