## 1. Reference

- [『ゼロから作るDeep Learning ❸ ―フレームワーク編』](https://bookmeter.com/books/15556819)

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

============================== 15 passed in 2.11s ==============================
```

## 4. Static Code Analysis

```command
$ flake8 .
$ autoflake8 --in-place --remove-duplicate-keys --remove-unused-variables --recursive .
$ autopep8 --in-place --aggressive --aggressive --recursive .
```

## 5. Type Checks

```command
$ for chapter in ./*/
$ do
$   echo "===== mypy ${chapter} ====="
$   mypy "${chapter}" || status=1
$ done
===== mypy ./1_basic_differential/ =====
Success: no issues found in 13 source files
===== mypy ./2_natural_coding/ =====
Success: no issues found in 3 source files
===== mypy ./3_higher differentiation/ =====
Success: no issues found in 3 source files
===== mypy ./4_neural_network/ =====
Success: no issues found in 3 source files
===== mypy ./5_de_zero/ =====
Success: no issues found in 3 source files
```
