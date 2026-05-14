from mul_layer import MulLayer


def test_forward():
    apple_layer = MulLayer()
    tax_layer = MulLayer()
    apple_price = apple_layer.forward(100, 2)
    price = tax_layer.forward(apple_price, 1.1)
    assert int(price) == 220


def test_backward():
    apple_layer = MulLayer()
    tax_layer = MulLayer()
    apple_price = apple_layer.forward(100, 2)
    tax_layer.forward(apple_price, 1.1)
    dapple_price, dtax = tax_layer.backward(1)
    dapple, dapple_num = apple_layer.backward(dapple_price)
    assert dapple == 2.2
    assert int(dapple_num) == 110
    assert dtax == 200
