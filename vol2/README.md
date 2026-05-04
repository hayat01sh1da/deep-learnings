## 1. Reference

- [『ゼロから作るDeep Learning ② ―自然言語処理編』](https://bookmeter.com/books/12738319)

## 2. Install Libraries via requirements.txt

```command
$ pip install -r requirements.txt
```

## 3. Unit Test

```command
$ pytest
====================================================================== test session starts =======================================================================
platform linux -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0
rootdir: /mnt/c/Users/binlh/Documents/development/deep-learnings/vol2
collected 153 items                                                                                                                                              

test/test_affine.py ..                                                                                                                                     [  1%]
test/test_attention.py ..                                                                                                                                  [  2%]
test/test_attention_decoder.py ...                                                                                                                         [  4%]
test/test_attention_encoder.py ..                                                                                                                          [  5%]
test/test_attention_weight.py ..                                                                                                                           [  7%]
test/test_better_rnnlm.py ....                                                                                                                             [  9%]
test/test_cbow.py ..                                                                                                                                       [ 11%]
test/test_count_based_methods.py ...........                                                                                                               [ 18%]
test/test_cross_entropy_error.py .                                                                                                                         [ 18%]
test/test_decoder.py ...                                                                                                                                   [ 20%]
test/test_embedding.py ....                                                                                                                                [ 23%]
test/test_embedding_dot.py ........                                                                                                                        [ 28%]
test/test_encoder.py ..                                                                                                                                    [ 30%]
test/test_lstm.py ...                                                                                                                                      [ 32%]
test/test_matmul.py ..                                                                                                                                     [ 33%]
test/test_matrix.py ........                                                                                                                               [ 38%]
test/test_negative_sampling_loss.py ....                                                                                                                   [ 41%]
test/test_neural_network.py ..                                                                                                                             [ 42%]
test/test_peeky_decoder.py ...                                                                                                                             [ 44%]
test/test_repeat.py ..                                                                                                                                     [ 45%]
test/test_rnn.py ..                                                                                                                                        [ 47%]
test/test_rnn_gen.py FF                                                                                                                                    [ 48%]
test/test_rnnlm.py .FF..F                                                                                                                                  [ 52%]
test/test_rnnlm_trainer.py ...                                                                                                                             [ 54%]
test/test_seq2seq.py ...                                                                                                                                   [ 56%]
test/test_sequence.py .....                                                                                                                                [ 59%]
test/test_sigmoid.py ..                                                                                                                                    [ 60%]
test/test_sigmoid_with_loss.py ..                                                                                                                          [ 62%]
test/test_simple_cbow.py ..                                                                                                                                [ 63%]
test/test_simple_rnnlm.py ....                                                                                                                             [ 66%]
test/test_simple_word2vec.py ...                                                                                                                           [ 67%]
test/test_softmax.py ...                                                                                                                                   [ 69%]
test/test_softmax_with_loss.py ..                                                                                                                          [ 71%]
test/test_spiral_dataset.py ...                                                                                                                            [ 73%]
test/test_sum.py ..                                                                                                                                        [ 74%]
test/test_time_affine.py ..                                                                                                                                [ 75%]
test/test_time_attention.py ..                                                                                                                             [ 77%]
test/test_time_dropout.py ..                                                                                                                               [ 78%]
test/test_time_embedding.py ..                                                                                                                             [ 79%]
test/test_time_lstm.py ...                                                                                                                                 [ 81%]
test/test_time_rnn.py ...                                                                                                                                  [ 83%]
test/test_time_softmax_with_loss.py ..                                                                                                                     [ 84%]
test/test_train_custom_loop.py ......                                                                                                                      [ 88%]
test/test_trainer.py ......                                                                                                                                [ 92%]
test/test_two_layer_net.py ...                                                                                                                             [ 94%]
test/test_unigram_sampler.py ..                                                                                                                            [ 96%]
test/test_vector.py ....                                                                                                                                   [ 98%]
test/test_weight_sum.py ..                                                                                                                                 [100%]

============================================================================ FAILURES ============================================================================
________________________________________________________________ TestRNNLMGen.test_generate_text _________________________________________________________________

self = <test_rnn_gen.TestRNNLMGen testMethod=test_generate_text>

    def test_generate_text(self):
>       word_ids = self.rnnlm_gen.word_ids_list(self.start_id, self.skip_ids)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

test/test_rnn_gen.py:32: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <rnnlm_gen.RNNLMGen object at 0x71e156625d30>, start_id = 316, skip_ids = [27, 26, 416], sample_size = 100

    def word_ids_list(self, start_id, skip_ids = None, sample_size = 100):
        word_ids = [start_id]
        x        = start_id
        while len(word_ids) < sample_size:
            x       = np.array(x).reshape(1, 1)
            score   = self._predict(x)
            p       = self.softmax.calc_softmax(score.flatten())
            sampled = np.random.choice(len(p), size = 1, p = p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
>               word_ids.append(int(x))
                                ^^^^^^
E               TypeError: only 0-dimensional arrays can be converted to Python scalars

src/rnnlm_gen.py:16: TypeError
---------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------
Warning: No file: ../pkl/better_rnnlm.pkl — creating placeholder params
________________________________________________________________ TestRNNLMGen.test_word_ids_list _________________________________________________________________

self = <test_rnn_gen.TestRNNLMGen testMethod=test_word_ids_list>

    def test_word_ids_list(self):
>       word_ids = self.rnnlm_gen.word_ids_list(self.start_id, self.skip_ids)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

test/test_rnn_gen.py:28: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <rnnlm_gen.RNNLMGen object at 0x71e15678b4d0>, start_id = 316, skip_ids = [27, 26, 416], sample_size = 100

    def word_ids_list(self, start_id, skip_ids = None, sample_size = 100):
        word_ids = [start_id]
        x        = start_id
        while len(word_ids) < sample_size:
            x       = np.array(x).reshape(1, 1)
            score   = self._predict(x)
            p       = self.softmax.calc_softmax(score.flatten())
            sampled = np.random.choice(len(p), size = 1, p = p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
>               word_ids.append(int(x))
                                ^^^^^^
E               TypeError: only 0-dimensional arrays can be converted to Python scalars

src/rnnlm_gen.py:16: TypeError
---------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------
Warning: No file: ../pkl/better_rnnlm.pkl — creating placeholder params
_____________________________________________________________________ TestRNNLM.test_forward _____________________________________________________________________

self = <test_rnnlm.TestRNNLM testMethod=test_forward>

    def test_forward(self):
        loss = self.rnnlm.forward(self.xs, self.ts)
>       self.assertEqual(round(loss, 2), 1.94)
E       AssertionError: np.float64(1.95) != 1.94

test/test_rnnlm.py:44: AssertionError
___________________________________________________________________ TestRNNLM.test_load_params ___________________________________________________________________

self = <test_rnnlm.TestRNNLM testMethod=test_load_params>

    def test_load_params(self):
>       self.rnnlm.load_params()

test/test_rnnlm.py:65: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <rnnlm.RNNLM object at 0x71e15678a850>, file_path = '../pkl/rnnlm.pkl'

    def load_params(self, file_path='../pkl/rnnlm.pkl'):
>       with open(file_path, 'rb') as f:
             ^^^^^^^^^^^^^^^^^^^^^
E       FileNotFoundError: [Errno 2] No such file or directory: '../pkl/rnnlm.pkl'

src/rnnlm.py:62: FileNotFoundError
___________________________________________________________________ TestRNNLM.test_save_params ___________________________________________________________________

self = <test_rnnlm.TestRNNLM testMethod=test_save_params>

    def test_save_params(self):
        self.rnnlm.forward(self.xs, self.ts)
        self.rnnlm.backward()
>       self.rnnlm.save_params()

test/test_rnnlm.py:61: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <rnnlm.RNNLM object at 0x71e1567b8710>, file_path = '../pkl/rnnlm.pkl'

    def save_params(self, file_path='../pkl/rnnlm.pkl'):
>       with open(file_path, 'wb') as f:
             ^^^^^^^^^^^^^^^^^^^^^
E       FileNotFoundError: [Errno 2] No such file or directory: '../pkl/rnnlm.pkl'

src/rnnlm.py:58: FileNotFoundError
==================================================================== short test summary info =====================================================================
FAILED test/test_rnn_gen.py::TestRNNLMGen::test_generate_text - TypeError: only 0-dimensional arrays can be converted to Python scalars
FAILED test/test_rnn_gen.py::TestRNNLMGen::test_word_ids_list - TypeError: only 0-dimensional arrays can be converted to Python scalars
FAILED test/test_rnnlm.py::TestRNNLM::test_forward - AssertionError: np.float64(1.95) != 1.94
FAILED test/test_rnnlm.py::TestRNNLM::test_load_params - FileNotFoundError: [Errno 2] No such file or directory: '../pkl/rnnlm.pkl'
FAILED test/test_rnnlm.py::TestRNNLM::test_save_params - FileNotFoundError: [Errno 2] No such file or directory: '../pkl/rnnlm.pkl'
================================================================= 5 failed, 148 passed in 23.03s =================================================================
```
