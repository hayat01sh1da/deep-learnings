import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable

def _numerical_gradient_1d(f: Callable[[NDArray[Any]], float], x: NDArray[Any]) -> NDArray[Any]:
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        # f(x+h)
        fxh1 = f(x)
        x[idx] = tmp_val - h
        # f(x-h)
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        # Revert the value
        x[idx] = tmp_val
    return grad

def numerical_gradient_2d(f: Callable[[NDArray[Any]], float], X: NDArray[Any]) -> NDArray[Any]:
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        return grad

def numerical_gradient(f: Callable[[NDArray[Any]], float], x: NDArray[Any]) -> NDArray[Any]:
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        # f(x+h)
        fxh1 = f(x)
        x[idx] = tmp_val - h
        # f(x-h)
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        # Revert the value
        x[idx] = tmp_val
        it.iternext()
    return grad
