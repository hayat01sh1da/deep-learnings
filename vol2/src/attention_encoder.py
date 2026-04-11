import numpy as np
import sys
from numpy.typing import NDArray
from typing import Any
sys.path.append('./layers')
from encoder import Encoder

class AttentionEncoder(Encoder):
    def forward(self, xs: NDArray[Any]) -> NDArray[Any]:
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs

    def backward(self, dhs: NDArray[Any]) -> NDArray[Any]:
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
