from simple_perceptron import SimplePerceptron

class MultiLayeredPerceptron(SimplePerceptron):
    def __init__(self) -> None:
        pass

    def xor_gate(self, x1: int, x2: int) -> int:
        s1 = self.nand_gate(x1, x2)
        s2 = self.or_gate(x1, x2)
        y  = self.and_gate(s1, s2)
        return y
