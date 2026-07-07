[![Actions Status: Vol.1](https://github.com/hayat01sh1da/deep-learnings/workflows/Vol.1/badge.svg)](https://github.com/hayat01sh1da/deep-learnings/actions?query=workflow%3A%22Vol.1%22)
[![Actions Status: Vol.2](https://github.com/hayat01sh1da/deep-learnings/workflows/Vol.2/badge.svg)](https://github.com/hayat01sh1da/deep-learnings/actions?query=workflow%3A%22Vol.2%22)
[![Actions Status: Vol.3](https://github.com/hayat01sh1da/deep-learnings/workflows/Vol.3/badge.svg)](https://github.com/hayat01sh1da/deep-learnings/actions?query=workflow%3A%22Vol.3%22)
[![Actions Status: Python - Daily Dependencies Update](https://github.com/hayat01sh1da/deep-learnings/workflows/Python%20-%20Daily%20Dependencies%20Update/badge.svg)](https://github.com/hayat01sh1da/deep-learnings/actions?query=workflow%3A%22Python%20-%20Daily%20Dependencies%20Update%22)
[![Actions Status: CodeQL](https://github.com/hayat01sh1da/deep-learnings/workflows/CodeQL/badge.svg)](https://github.com/hayat01sh1da/deep-learnings/actions?query=workflow%3A%22CodeQL%22)

## 1. Common Environment

- WSL (Ubuntu 25.10)
- Python 3.14.6
- pip 26.1.2

## 2. READMEs

- [Vol.1](./vol1/README.md)
- [Vol.2](./vol2/README.md)
- [Vol.3](./vol3/README.md)

## 3. Unit Tests

```command
$ for chapter in ./vol*/
$ do
$   echo "===== mypy ${chapter} ====="
$   cd "${chapter}"
$   pytest .
$ done
===== mypy ./vol1/ =====
============================= test session starts ==============================
platform linux -- Python 3.14.6, pytest-9.0.3, pluggy-1.6.0
rootdir: /mnt/c/Users/binlh/Development/personal/deep-learnings
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

============================== 48 passed in 1.97s ==============================
===== mypy ./vol2/ =====
============================= test session starts ==============================
platform linux -- Python 3.14.6, pytest-9.0.3, pluggy-1.6.0
rootdir: /mnt/c/Users/binlh/Development/personal/deep-learnings
configfile: pyproject.toml
collected 153 items

test/test_affine.py ..                                                   [  1%]
test/test_attention.py ..                                                [  2%]
test/test_attention_decoder.py ...                                       [  4%]
test/test_attention_encoder.py ..                                        [  5%]
test/test_attention_weight.py ..                                         [  7%]
test/test_better_rnnlm.py ....                                           [  9%]
test/test_cbow.py ..                                                     [ 11%]
test/test_count_based_methods.py ...........                             [ 18%]
test/test_cross_entropy_error.py .                                       [ 18%]
test/test_decoder.py ...                                                 [ 20%]
test/test_embedding.py ....                                              [ 23%]
test/test_embedding_dot.py ........                                      [ 28%]
test/test_encoder.py ..                                                  [ 30%]
test/test_lstm.py ...                                                    [ 32%]
test/test_matmul.py ..                                                   [ 33%]
test/test_matrix.py ........                                             [ 38%]
test/test_negative_sampling_loss.py ....                                 [ 41%]
test/test_neural_network.py ..                                           [ 42%]
test/test_peeky_decoder.py ...                                           [ 44%]
test/test_repeat.py ..                                                   [ 45%]
test/test_rnn.py ..                                                      [ 47%]
test/test_rnn_gen.py ..                                                  [ 48%]
test/test_rnnlm.py ......                                                [ 52%]
test/test_rnnlm_trainer.py ...                                           [ 54%]
test/test_seq2seq.py ...                                                 [ 56%]
test/test_sequence.py .....                                              [ 59%]
test/test_sigmoid.py ..                                                  [ 60%]
test/test_sigmoid_with_loss.py ..                                        [ 62%]
test/test_simple_cbow.py ..                                              [ 63%]
test/test_simple_rnnlm.py ....                                           [ 66%]
test/test_simple_word2vec.py ...                                         [ 67%]
test/test_softmax.py ...                                                 [ 69%]
test/test_softmax_with_loss.py ..                                        [ 71%]
test/test_spiral_dataset.py ...                                          [ 73%]
test/test_sum.py ..                                                      [ 74%]
test/test_time_affine.py ..                                              [ 75%]
test/test_time_attention.py ..                                           [ 77%]
test/test_time_dropout.py ..                                             [ 78%]
test/test_time_embedding.py ..                                           [ 79%]
test/test_time_lstm.py ...                                               [ 81%]
test/test_time_rnn.py ...                                                [ 83%]
test/test_time_softmax_with_loss.py ..                                   [ 84%]
test/test_train_custom_loop.py ......                                    [ 88%]
test/test_trainer.py ......                                              [ 92%]
test/test_two_layer_net.py ...                                           [ 94%]
test/test_unigram_sampler.py ..                                          [ 96%]
test/test_vector.py ....                                                 [ 98%]
test/test_weight_sum.py ..                                               [100%]

============================= 153 passed in 31.00s =============================
===== mypy ./vol3/ =====
============================= test session starts ==============================
platform linux -- Python 3.14.6, pytest-9.0.3, pluggy-1.6.0
rootdir: /mnt/c/Users/binlh/Development/personal/deep-learnings
configfile: pyproject.toml
collected 15 items

1_basic_differential/test/test_diff.py ..                                [ 13%]
1_basic_differential/test/test_exp.py ..                                 [ 26%]
1_basic_differential/test/test_function.py .                             [ 33%]
1_basic_differential/test/test_square.py ..                              [ 46%]
1_basic_differential/test/test_template.py .                             [ 53%]
1_basic_differential/test/test_variable.py ...                           [ 73%]
2_natural_coding/test/test_template.py .                                 [ 80%]
3_higher differentiation/test/test_template.py .                         [ 86%]
4_neural_network/test/test_template.py .                                 [ 93%]
5_de_zero/test/test_template.py .                                        [100%]

============================== 15 passed in 2.02s ==============================

```
