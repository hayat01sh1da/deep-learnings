import os

import pytest

from spiral_dataset import SpiralDataset


@pytest.fixture
def spiral_dataset():
    return SpiralDataset()


def test_x_shape(spiral_dataset):
    assert spiral_dataset.x.shape == (300, 2)


def test_t_shape(spiral_dataset):
    assert spiral_dataset.t.shape == (300, 3)


def test_save_plot_image(spiral_dataset):
    file_path = '../img/spiral_plot.png'
    spiral_dataset.save_plot_image(file_path)
    assert os.path.exists(file_path)
