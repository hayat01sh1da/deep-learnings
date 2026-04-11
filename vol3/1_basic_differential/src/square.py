from typing import Any
from function import Function

class Square(Function):
    def forward(self, x: Any) -> Any:
        return x ** 2
