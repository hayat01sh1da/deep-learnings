from function import Function

class Square(Function):
    def forward(self, x):
        return x ** 2
