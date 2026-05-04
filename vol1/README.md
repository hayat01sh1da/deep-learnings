## 1. Reference

- [『ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装』](https://bookmeter.com/books/11128002)

## 2. Install Libraries via requirements.txt

```command
$ pip install -r requirements.txt
```

## 3. Unit Test

```command
$ pytest
====================================================================== test session starts =======================================================================
platform linux -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0
rootdir: /mnt/c/Users/binlh/Documents/development/deep-learnings/vol1
collected 32 items                                                                                                                                               

test/test_add_layer.py ..                                                                                                                                  [  6%]
test/test_affine.py ..                                                                                                                                     [ 12%]
test/test_mul_layer.py ..                                                                                                                                  [ 18%]
test/test_multi_layerred_perceptron.py .                                                                                                                   [ 21%]
test/test_neural_network.py ...                                                                                                                            [ 31%]
test/test_neural_network_clean.py ...                                                                                                                      [ 40%]
test/test_neural_network_learning.py ..........                                                                                                            [ 71%]
test/test_relu.py ..                                                                                                                                       [ 78%]
test/test_sigmoid.py ..                                                                                                                                    [ 84%]
test/test_simple_perceptron.py ...                                                                                                                         [ 93%]
test/test_softmax_with_loss.py ..                                                                                                                          [100%]

======================================================================= 32 passed in 1.62s =======================================================================
```
