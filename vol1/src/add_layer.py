class AddLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x: float, y: float) -> float:
        out = x + y
        return out

    def backward(self, dout: float) -> tuple[float, float]:
        dx = dout * 1
        dy = dout * 1
        return dx, dy
