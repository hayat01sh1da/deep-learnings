## 1. Reference

- [『ゼロから作るDeep Learning ② ―自然言語処理編』](https://bookmeter.com/books/12738319)

## 2. Install Libraries via requirements.txt

```command
$ pip install -r requirements.txt
```

## 3. Unit Test

```command
$ pytest .
============================= test session starts ==============================
platform linux -- Python 3.14.6, pytest-9.0.3, pluggy-1.6.0
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

$ pytest .
============================= test session starts ==============================
platform linux -- Python 3.14.6, pytest-9.0.3, pluggy-1.6.0
rootdir: deep-learnings
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

============================= 153 passed in 29.77s =============================
```

## 4. Static Code Analysis

```command
$ flake8 .
./src/concerns/eval_perplexity.py:9:17: F841 local variable 'loss_cnt' is assigned to but never used
./src/layers/attention_weight.py:26:9: F841 local variable 'dh' is assigned to but never used
./src/layers/sigmoid_with_loss.py:1:1: F403 'from cross_entropy_error import *' used; unable to detect undefined names
./src/layers/sigmoid_with_loss.py:17:16: F405 'cross_entropy_error' may be undefined, or defined from star imports: cross_entropy_error
./src/layers/softmax_with_loss.py:1:1: F403 'from cross_entropy_error import *' used; unable to detect undefined names
./src/layers/softmax_with_loss.py:21:16: F405 'cross_entropy_error' may be undefined, or defined from star imports: cross_entropy_error
./src/models/base_model.py:29:80: E501 line too long (104 > 79 characters)
./src/models/two_layer_net.py:11:9: E741 ambiguous variable name 'I'
./src/models/two_layer_net.py:13:9: E741 ambiguous variable name 'O'
./src/plot_shim.py:4:80: E501 line too long (81 > 79 characters)
./src/plot_shim.py:5:80: E501 line too long (86 > 79 characters)
./src/rnnlm_trainer.py:4:1: F403 'from clip_grads import *' used; unable to detect undefined names
./src/rnnlm_trainer.py:18:80: E501 line too long (92 > 79 characters)
./src/rnnlm_trainer.py:38:80: E501 line too long (102 > 79 characters)
./src/rnnlm_trainer.py:58:13: F402 import 'time' from line 2 shadowed by loop variable
./src/rnnlm_trainer.py:91:21: F405 'clip_grads' may be undefined, or defined from star imports: clip_grads
./src/rnnlm_trainer.py:101:80: E501 line too long (85 > 79 characters)
./src/rnnlm_trainer.py:102:80: E501 line too long (90 > 79 characters)
./src/spiral_dataset.py:2:1: F403 'from spiral import *' used; unable to detect undefined names
./src/spiral_dataset.py:9:26: F405 'load_data' may be undefined, or defined from star imports: spiral
./src/spiral_dataset.py:18:80: E501 line too long (85 > 79 characters)
./src/spiral_dataset.py:19:80: E501 line too long (85 > 79 characters)
./src/train_custom_loop.py:68:80: E501 line too long (99 > 79 characters)
./src/train_custom_loop.py:105:80: E501 line too long (80 > 79 characters)
./src/train_custom_loop.py:106:80: E501 line too long (80 > 79 characters)
./src/trainer.py:1:1: F403 'from clip_grads import *' used; unable to detect undefined names
./src/trainer.py:28:80: E501 line too long (92 > 79 characters)
./src/trainer.py:48:80: E501 line too long (102 > 79 characters)
./src/trainer.py:71:80: E501 line too long (83 > 79 characters)
./src/trainer.py:72:80: E501 line too long (80 > 79 characters)
./src/trainer.py:98:21: F405 'clip_grads' may be undefined, or defined from star imports: clip_grads
./src/trainer.py:105:80: E501 line too long (92 > 79 characters)
./test/test_count_based_methods.py:30:80: E501 line too long (83 > 79 characters)
./test/test_count_based_methods.py:150:80: E501 line too long (119 > 79 characters)
./test/test_count_based_methods.py:151:80: E501 line too long (83 > 79 characters)
./test/test_count_based_methods.py:152:80: E501 line too long (130 > 79 characters)
./test/test_count_based_methods.py:153:80: E501 line too long (127 > 79 characters)
./test/test_count_based_methods.py:154:80: E501 line too long (129 > 79 characters)
./test/test_count_based_methods.py:155:80: E501 line too long (128 > 79 characters)
./test/test_count_based_methods.py:156:80: E501 line too long (130 > 79 characters)
./test/test_lstm.py:13:80: E501 line too long (85 > 79 characters)
./test/test_lstm.py:17:80: E501 line too long (103 > 79 characters)
./test/test_lstm.py:19:80: E501 line too long (119 > 79 characters)
./test/test_lstm.py:23:80: E501 line too long (89 > 79 characters)
./test/test_lstm.py:24:80: E501 line too long (89 > 79 characters)
./test/test_lstm.py:25:80: E501 line too long (89 > 79 characters)
./test/test_lstm.py:26:80: E501 line too long (89 > 79 characters)
./test/test_lstm.py:27:80: E501 line too long (89 > 79 characters)
./test/test_lstm.py:28:80: E501 line too long (89 > 79 characters)
./test/test_lstm.py:29:80: E501 line too long (89 > 79 characters)
./test/test_lstm.py:30:80: E501 line too long (121 > 79 characters)
./test/test_lstm.py:31:80: E501 line too long (89 > 79 characters)
./test/test_lstm.py:32:80: E501 line too long (91 > 79 characters)
./test/test_lstm.py:34:80: E501 line too long (157 > 79 characters)
./test/test_lstm.py:35:80: E501 line too long (160 > 79 characters)
./test/test_lstm.py:36:80: E501 line too long (158 > 79 characters)
./test/test_lstm.py:39:80: E501 line too long (80 > 79 characters)
./test/test_lstm.py:40:80: E501 line too long (80 > 79 characters)
./test/test_time_attention.py:9:80: E501 line too long (80 > 79 characters)
./test/test_time_lstm.py:11:80: E501 line too long (85 > 79 characters)
./test/test_time_lstm.py:15:80: E501 line too long (103 > 79 characters)
./test/test_time_lstm.py:17:80: E501 line too long (119 > 79 characters)
./test/test_time_lstm.py:21:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:22:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:23:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:24:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:25:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:26:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:27:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:28:80: E501 line too long (121 > 79 characters)
./test/test_time_lstm.py:29:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:30:80: E501 line too long (91 > 79 characters)
./test/test_time_lstm.py:32:80: E501 line too long (157 > 79 characters)
./test/test_time_lstm.py:33:80: E501 line too long (160 > 79 characters)
./test/test_time_lstm.py:34:80: E501 line too long (158 > 79 characters)
./test/test_time_lstm.py:37:80: E501 line too long (80 > 79 characters)
./test/test_time_lstm.py:38:80: E501 line too long (80 > 79 characters)
./test/test_time_lstm.py:43:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:44:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:45:80: E501 line too long (90 > 79 characters)
./test/test_time_lstm.py:46:80: E501 line too long (90 > 79 characters)
./test/test_time_lstm.py:47:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:48:80: E501 line too long (87 > 79 characters)
./test/test_time_lstm.py:51:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:52:80: E501 line too long (86 > 79 characters)
./test/test_time_lstm.py:53:80: E501 line too long (92 > 79 characters)
./test/test_time_lstm.py:54:80: E501 line too long (91 > 79 characters)
./test/test_time_lstm.py:55:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:56:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:59:80: E501 line too long (87 > 79 characters)
./test/test_time_lstm.py:60:80: E501 line too long (87 > 79 characters)
./test/test_time_lstm.py:61:80: E501 line too long (92 > 79 characters)
./test/test_time_lstm.py:62:80: E501 line too long (89 > 79 characters)
./test/test_time_lstm.py:63:80: E501 line too long (91 > 79 characters)
./test/test_time_lstm.py:64:80: E501 line too long (92 > 79 characters)
./test/test_time_lstm.py:72:80: E501 line too long (83 > 79 characters)
./test/test_time_lstm.py:73:80: E501 line too long (83 > 79 characters)
./test/test_time_lstm.py:74:80: E501 line too long (82 > 79 characters)
./test/test_time_lstm.py:75:80: E501 line too long (80 > 79 characters)
./test/test_time_lstm.py:76:80: E501 line too long (83 > 79 characters)
./test/test_time_lstm.py:77:80: E501 line too long (83 > 79 characters)
./test/test_time_lstm.py:80:80: E501 line too long (88 > 79 characters)
./test/test_time_lstm.py:81:80: E501 line too long (83 > 79 characters)
./test/test_time_lstm.py:82:80: E501 line too long (82 > 79 characters)
./test/test_time_lstm.py:83:80: E501 line too long (82 > 79 characters)
./test/test_time_lstm.py:84:80: E501 line too long (86 > 79 characters)
./test/test_time_lstm.py:85:80: E501 line too long (84 > 79 characters)
./test/test_time_lstm.py:88:80: E501 line too long (84 > 79 characters)
./test/test_time_lstm.py:89:80: E501 line too long (84 > 79 characters)
./test/test_time_lstm.py:90:80: E501 line too long (83 > 79 characters)
./test/test_time_lstm.py:91:80: E501 line too long (87 > 79 characters)
./test/test_time_lstm.py:92:80: E501 line too long (83 > 79 characters)
./test/test_time_lstm.py:93:80: E501 line too long (83 > 79 characters)
./test/test_time_lstm.py:99:80: E501 line too long (86 > 79 characters)
./test/test_time_lstm.py:100:80: E501 line too long (85 > 79 characters)
./test/test_time_lstm.py:101:80: E501 line too long (87 > 79 characters)
./test/test_time_lstm.py:102:80: E501 line too long (85 > 79 characters)
./test/test_time_lstm.py:103:80: E501 line too long (86 > 79 characters)
./test/test_time_lstm.py:104:80: E501 line too long (86 > 79 characters)
./test/test_time_lstm.py:107:80: E501 line too long (85 > 79 characters)
./test/test_time_lstm.py:108:80: E501 line too long (87 > 79 characters)
./test/test_time_lstm.py:109:80: E501 line too long (84 > 79 characters)
./test/test_time_lstm.py:110:80: E501 line too long (86 > 79 characters)
./test/test_time_lstm.py:111:80: E501 line too long (84 > 79 characters)
./test/test_time_lstm.py:112:80: E501 line too long (85 > 79 characters)
./test/test_time_lstm.py:115:80: E501 line too long (84 > 79 characters)
./test/test_time_lstm.py:116:80: E501 line too long (82 > 79 characters)
./test/test_time_lstm.py:117:80: E501 line too long (85 > 79 characters)
./test/test_time_lstm.py:118:80: E501 line too long (84 > 79 characters)
./test/test_time_lstm.py:119:80: E501 line too long (85 > 79 characters)
./test/test_time_lstm.py:120:80: E501 line too long (86 > 79 characters)
./test/test_time_softmax_with_loss.py:35:80: E501 line too long (128 > 79 characters)
./test/test_time_softmax_with_loss.py:36:80: E501 line too long (130 > 79 characters)
./test/test_time_softmax_with_loss.py:37:80: E501 line too long (128 > 79 characters)
./test/test_trainer.py:80:80: E501 line too long (89 > 79 characters)
./test/test_trainer.py:81:80: E501 line too long (91 > 79 characters)
./train/train_seq2seq.py:16:80: E501 line too long (82 > 79 characters)
$ autoflake8 --in-place --remove-duplicate-keys --remove-unused-variables --recursive .
$ autopep8 --in-place --aggressive --aggressive --recursive .
```

## 5. Type Checks

```command
$ mypy .
Success: no issues found in 120 source files
```
