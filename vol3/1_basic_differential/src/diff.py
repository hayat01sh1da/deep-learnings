from exp import Exp
from square import Square
from variable import Variable

def numerical_diff(f: type, x: Variable, eps: float = 1e-4) -> float:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def f(x: Variable) -> Variable:
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))
