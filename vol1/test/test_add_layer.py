import pytest

from add_layer import AddLayer
from mul_layer import MulLayer


@pytest.fixture
def layers():
    return {
        'apple': MulLayer(),
        'orange': MulLayer(),
        'apple_orange': AddLayer(),
        'tax': MulLayer(),
    }


APPLE = 100
APPLE_NUM = 2
ORANGE = 150
ORANGE_NUM = 3
TAX = 1.1


def test_forward(layers):
    apple_price = layers['apple'].forward(APPLE, APPLE_NUM)
    orange_price = layers['orange'].forward(ORANGE, ORANGE_NUM)
    apple_orange_price = layers['apple_orange'].forward(
        apple_price, orange_price)
    price = layers['tax'].forward(apple_orange_price, TAX)
    assert int(price) == 715


def test_backward(layers):
    apple_price = layers['apple'].forward(APPLE, APPLE_NUM)
    orange_price = layers['orange'].forward(ORANGE, ORANGE_NUM)
    apple_orange_price = layers['apple_orange'].forward(
        apple_price, orange_price)
    layers['tax'].forward(apple_orange_price, TAX)

    dall_price, dtax = layers['tax'].backward(1)
    dapple_price, dorange_price = layers['apple_orange'].backward(dall_price)
    dorange, dorange_num = layers['orange'].backward(dorange_price)
    dapple, dapple_num = layers['apple'].backward(dapple_price)

    assert dapple == 2.2
    assert int(dapple_num) == 110
    assert float(f'{dorange:.1f}') == 3.3
    assert int(dorange_num) == 165
    assert dtax == 650
