## 1. Reference

- [『ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装』](https://bookmeter.com/books/11128002)

## 2. Install Libraries via requirements.txt

```command
$ pip install -r requirements.txt
```

## 3. Unit Test

```command
$ invoke
============================= test session starts ==============================
platform linux -- Python 3.14.5, pytest-9.0.3, pluggy-1.6.0
rootdir: deep-learnings
configfile: pyproject.toml
collected 48 items

test/test_add_layer.py ..                                                [  4%]
test/test_affine.py ..                                                   [  8%]
test/test_mul_layer.py ..                                                [ 12%]
test/test_multi_layerred_perceptron.py ....                              [ 20%]
test/test_neural_network.py ...                                          [ 27%]
test/test_neural_network_clean.py ...                                    [ 33%]
test/test_neural_network_learning.py ..............                      [ 62%]
test/test_relu.py ..                                                     [ 66%]
test/test_sigmoid.py ..                                                  [ 70%]
test/test_simple_perceptron.py ............                              [ 95%]
test/test_softmax_with_loss.py ..                                        [100%]

============================== 48 passed in 4.02s ==============================
```

## 4. Static Code Analysis

```command
$ flake8 .
./dataset/mnist.py:40:80: E501 line too long (101 > 79 characters)
./src/adam.py:21:80: E501 line too long (84 > 79 characters)
./src/neural_network.py:21:80: E501 line too long (84 > 79 characters)
./src/neural_network.py:22:80: E501 line too long (81 > 79 characters)
$ autoflake8 --in-place --remove-duplicate-keys --remove-unused-variables --recursive .
$ autopep8 --in-place --aggressive --aggressive --recursive .
```

## 5. Type Checks

```command
$ mypy .
Success: no issues found in 42 source files
```
