## 1. Common Environment

- WSL(Ubuntu 24.04.1 LTS)
- Python 3.13.7

## 2. READMEs

- [Vol.1](./vol1/README.md)
- [Vol.2](./vol2/README.md)
- [Vol.3](./vol3/README.md)

## 3. Bulk Execution of Unit Tests

```command
$ bash run_unittests.sh 
.......EE.........
======================================================================
ERROR: test_neural_network (unittest.loader._FailedTest.test_neural_network)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network.py", line 11, in <module>
    from neural_network import NeuralNetwork
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network.py", line 5, in <module>
    from mnist import load_mnist
ModuleNotFoundError: No module named 'mnist'


======================================================================
ERROR: test_neural_network_learning (unittest.loader._FailedTest.test_neural_network_learning)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network_learning
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network_learning.py", line 9, in <module>
    from neural_network_learning import NeuralNetworkLearning
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network_learning.py", line 4, in <module>
    from dataset.mnist import load_mnist
ModuleNotFoundError: No module named 'dataset'


----------------------------------------------------------------------
Ran 18 tests in 0.207s

FAILED (errors=2)
run_unittests.sh: line 5: cd: vol2/: No such file or directory
.......EE.........
======================================================================
ERROR: test_neural_network (unittest.loader._FailedTest.test_neural_network)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network.py", line 11, in <module>
    from neural_network import NeuralNetwork
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network.py", line 5, in <module>
    from mnist import load_mnist
ModuleNotFoundError: No module named 'mnist'


======================================================================
ERROR: test_neural_network_learning (unittest.loader._FailedTest.test_neural_network_learning)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network_learning
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network_learning.py", line 9, in <module>
    from neural_network_learning import NeuralNetworkLearning
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network_learning.py", line 4, in <module>
    from dataset.mnist import load_mnist
ModuleNotFoundError: No module named 'dataset'


----------------------------------------------------------------------
Ran 18 tests in 0.199s

FAILED (errors=2)
run_unittests.sh: line 5: cd: vol3/: No such file or directory
.......EE.........
======================================================================
ERROR: test_neural_network (unittest.loader._FailedTest.test_neural_network)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network.py", line 11, in <module>
    from neural_network import NeuralNetwork
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network.py", line 5, in <module>
    from mnist import load_mnist
ModuleNotFoundError: No module named 'mnist'


======================================================================
ERROR: test_neural_network_learning (unittest.loader._FailedTest.test_neural_network_learning)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network_learning
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network_learning.py", line 9, in <module>
    from neural_network_learning import NeuralNetworkLearning
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network_learning.py", line 4, in <module>
    from dataset.mnist import load_mnist
ModuleNotFoundError: No module named 'dataset'


----------------------------------------------------------------------
Ran 18 tests in 0.142s

FAILED (errors=2)
.......EE.........
======================================================================
ERROR: test_neural_network (unittest.loader._FailedTest.test_neural_network)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network.py", line 11, in <module>
    from neural_network import NeuralNetwork
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network.py", line 5, in <module>
    from mnist import load_mnist
ModuleNotFoundError: No module named 'mnist'


======================================================================
ERROR: test_neural_network_learning (unittest.loader._FailedTest.test_neural_network_learning)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network_learning
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network_learning.py", line 9, in <module>
    from neural_network_learning import NeuralNetworkLearning
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network_learning.py", line 4, in <module>
    from dataset.mnist import load_mnist
ModuleNotFoundError: No module named 'dataset'


----------------------------------------------------------------------
Ran 18 tests in 0.182s

FAILED (errors=2)
.......EE.........
======================================================================
ERROR: test_neural_network (unittest.loader._FailedTest.test_neural_network)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network.py", line 11, in <module>
    from neural_network import NeuralNetwork
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network.py", line 5, in <module>
    from mnist import load_mnist
ModuleNotFoundError: No module named 'mnist'


======================================================================
ERROR: test_neural_network_learning (unittest.loader._FailedTest.test_neural_network_learning)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network_learning
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network_learning.py", line 9, in <module>
    from neural_network_learning import NeuralNetworkLearning
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network_learning.py", line 4, in <module>
    from dataset.mnist import load_mnist
ModuleNotFoundError: No module named 'dataset'


----------------------------------------------------------------------
Ran 18 tests in 0.197s

FAILED (errors=2)
.......EE.........
======================================================================
ERROR: test_neural_network (unittest.loader._FailedTest.test_neural_network)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network.py", line 11, in <module>
    from neural_network import NeuralNetwork
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network.py", line 5, in <module>
    from mnist import load_mnist
ModuleNotFoundError: No module named 'mnist'


======================================================================
ERROR: test_neural_network_learning (unittest.loader._FailedTest.test_neural_network_learning)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_neural_network_learning
Traceback (most recent call last):
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 396, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/hayat01sh1da/.pyenv/versions/3.13.0/lib/python3.13/unittest/loader.py", line 339, in _get_module_from_name
    __import__(name)
    ~~~~~~~~~~^^^^^^
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/test/test_neural_network_learning.py", line 9, in <module>
    from neural_network_learning import NeuralNetworkLearning
  File "/mnt/c/Users/binlh/Documents/data-science/deep-learnings/vol1/src/neural_network_learning.py", line 4, in <module>
    from dataset.mnist import load_mnist
ModuleNotFoundError: No module named 'dataset'


----------------------------------------------------------------------
Ran 18 tests in 0.192s

FAILED (errors=2)
```
