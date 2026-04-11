class MulLayer:
    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x: float, y: float) -> float:
        self.x = x
        self.y = y
        out    = x * y
        return out

    def backward(self, dout: float) -> tuple[float, float]:
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
