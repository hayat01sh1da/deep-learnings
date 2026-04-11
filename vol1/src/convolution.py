import numpy as np
from numpy.typing import NDArray
from typing import Any
import sys
sys.path.append('./lib')
from util import im2col

class Convolution:
    def __init__(self, W: NDArray[Any], b: NDArray[Any], stride: int = 1, pad: int = 0) -> None:
         self.W      = W
         self.b      = b
         self.stride = stride
         self.pad    = pad

    def forward(self, x: NDArray[Any]) -> NDArray[Any]:
        FN, C, FH  = self.W.shape
        N, C, H, W = x.shape
        out_h      = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w      = int(1 + (W + 2 * self.pad - FW) / self.stride)
        col        = im2col(x, FH, FW, self.stride, self.pad)
        col_W      = self.W.reshape(FN, -1).T
        out        = np.dot(col_w) + self.b
        out        = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out
