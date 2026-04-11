from typing import Any
from function import Function
import numpy as np

class Exp(Function):
    def forward(self, x: Any) -> Any:
        return np.exp(x)
