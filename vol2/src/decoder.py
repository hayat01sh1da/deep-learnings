import numpy as np
import sys
from numpy.typing import NDArray
from typing import Any
sys.path.append('./layers')
from time_embedding import TimeEmbedding
from time_lstm import TimeLSTM
from time_affine import TimeAffine

class Decoder:
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int) -> None:
        V           = vocab_size
        D           = wordvec_size
        H           = hidden_size
        rn          = np.random.randn
        embed_w     = (rn(V, D) / 100).astype('f')
        lstm_Wx     = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh     = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b      = np.zeros(4 * H).astype('f')
        affine_W    = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b    = np.zeros(V).astype('f')
        self.embed  = TimeEmbedding(embed_w)
        self.lstm   = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful = True)
        self.affine = TimeAffine(affine_W, affine_b)
        self.params = []
        self.grads  = []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads  += layer.grads

    def forward(self, xs: NDArray[Any], h: NDArray[Any]) -> NDArray[Any]:
        self.lstm.set_state(h)
        out   = self.embed.forward(xs)
        out   = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore: NDArray[Any]) -> NDArray[Any]:
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh

    def generate(self, h: NDArray[Any], start_id: int, sample_size: int) -> list[int]:
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)
        for _ in range(sample_size):
            x         = np.array(sample_id).reshape((1, 1))
            out       = self.embed.forward(x)
            out       = self.lstm.forward(out)
            score     = self.affine.forward(out)
            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))
        return sampled
