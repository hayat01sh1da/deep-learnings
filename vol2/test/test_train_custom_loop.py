import unittest
from os import path
import sys
import os
import shutil
import glob
sys.path.append('./src')
sys.path.append('./src/concerns')
sys.path.append('./src/layers')
sys.path.append('./src/models')
sys.path.append('./src/optimisers')
from train_custom_loop import TrainCustomLoop
from spiral import *

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.train_custom_loop = TrainCustomLoop()
        self.x, self.t         = load_data()
        self.max_epoch         = 300
        self.batch_size        = 30
        self.batch_x           = self.x[1 * self.batch_size: (1 + 1) * self.batch_size]
        self.batch_t           = self.t[1 * self.batch_size: (1 + 1) * self.batch_size]
        self.total_loss        = 0
        self.loss_count        = 0
        self.pycaches          = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_shuffle_data(self):
        xx, tt = self.train_custom_loop._shuffle_data(self.x, self.t)
        self.assertEqual(xx.shape, (300, 2))
        self.assertEqual(tt.shape, (300, 3))

    def test_update_params_with_grads(self):
        loss = self.train_custom_loop._update_params_with_grads(self.batch_x, self.batch_t, self.total_loss, self.loss_count)
        self.assertEqual(loss, 1.1074495352433567)

    def test_learning_process(self):
        loss             = self.train_custom_loop._update_params_with_grads(self.batch_x, self.batch_t, self.total_loss, self.loss_count)
        self.total_loss += loss
        self.loss_count += 1
        epoch            = 9
        iters            = 9
        data_size        = len(self.x)
        max_iters        = data_size // self.batch_size
        *_, process      = self.train_custom_loop._learning_process(self.total_loss, self.loss_count, epoch, iters, max_iters)
        self.assertEqual(process, '| epoch 10 | iter 10 / 10 | loss 1.07')

    def test_update(self):
        loss_list = self.train_custom_loop.update(self.x, self.t, self.max_epoch, self.batch_size)
        self.assertEqual(len(loss_list), 300)

    def test_save_plot_image(self):
        self.train_custom_loop.update(self.x, self.t, self.max_epoch, self.batch_size)
        file_path = '../img/train_custom_loop_plot.png'
        self.train_custom_loop.save_plot_image(file_path)
        self.assertTrue(os.path.exists(file_path))

    def test_save_dicision_boundary_image(self):
        self.train_custom_loop.update(self.x, self.t, self.max_epoch, self.batch_size)
        file_path = '../img/dicision_boundary.png'
        self.train_custom_loop.save_dicision_boundary_image(self.x, self.t, file_path)
        self.assertTrue(os.path.exists(file_path))

if __name__ == '__main__':
    unittest.main()
