import numpy as np
from numpy.typing import NDArray
from typing import Any

def clip_grads(grads: list[NDArray[Any]], max_norm: float) -> None:
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    rate       = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
