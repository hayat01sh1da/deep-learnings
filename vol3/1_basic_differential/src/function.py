from typing import Any
from variable import Variable

class Function:
    def __init__(self) -> None:
        pass

    def __call__(self, input: Variable) -> Variable:
        x      = input.data
        y      = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x: Any) -> Any:
        raise NotImplementedError()
