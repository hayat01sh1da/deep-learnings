import pytest

from multi_layered_perceptron import MultiLayeredPerceptron


@pytest.mark.parametrize(
    ('a', 'b', 'expected'),
    [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)],
)
def test_xor_gate(a, b, expected):
    assert MultiLayeredPerceptron().xor_gate(a, b) == expected
