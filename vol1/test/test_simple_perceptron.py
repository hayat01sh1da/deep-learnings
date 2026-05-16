import pytest

from simple_perceptron import SimplePerceptron


@pytest.fixture
def sp():
    return SimplePerceptron()


@pytest.mark.parametrize(('a', 'b', 'expected'),
                         [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)])
def test_and_gate(sp, a, b, expected):
    assert sp.and_gate(a, b) == expected


@pytest.mark.parametrize(('a', 'b', 'expected'),
                         [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)])
def test_nand_gate(sp, a, b, expected):
    assert sp.nand_gate(a, b) == expected


@pytest.mark.parametrize(('a', 'b', 'expected'),
                         [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)])
def test_or_gate(sp, a, b, expected):
    assert sp.or_gate(a, b) == expected
