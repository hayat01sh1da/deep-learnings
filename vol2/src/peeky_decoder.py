import numpy as np
import sys
sys.path.append('./layers')
from time_embedding import TimeEmbedding
from time_lstm import TimeLSTM
from time_affine import TimeAffine

class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V           = vocab_size
        D           = wordvec_size
        H           = hidden_size
        rn          = np.random.randn
        embed_w     = (rn(V, D) / 100).astype('f')
        lstm_Wx     = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh     = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b      = np.zeros(4 * H).astype('f')
        affine_W    = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b    = np.zeros(V).astype('f')
        self.embed  = TimeEmbedding(embed_w)
        self.lstm   = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful = True)
        self.affine = TimeAffine(affine_W, affine_b)
        self.params = []
        self.grads  = []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads  += layer.grads
        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape
        self.lstm.set_state(h)
        out        = self.embed.forward(xs)
        hs         = np.repeat(h, T, axis = 0).reshape(N, T, H)
        out        = np.concatenate((hs, out), axis = 2)
        out        = self.lstm.forward(out)
        out        = np.concatenate((hs, out), axis = 2)
        score      = self.affine.forward(out)
        self.cache = H
        return score

    def backward(self, dscore):
        H      = self.cache
        dout   = self.affine.backward(dscore)
        dout   = dout[:, :, H:]
        dhs0   = dout[:, :, :H]
        dout   = self.lstm.backward(dout)
        dembed = dout[:, :, H:]
        dhs1   = dout[:, :, :H]
        self.embed.backward(dembed)
        dhs = dhs0 + dhs1
        dh  = self.lstm.dh + np.sum(dhs, axis = 1)
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)
        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x       = np.array([char_id]).reshape((1, 1))
            out     = self.embed.forward(x)
            out     = np.concatenate((peeky_h, out), axis = 2)
            out     = self.lstm.forward(out)
            out     = np.concatenate((peeky_h, out), axis = 2)
            score   = self.affine.forward(out)
            char_id = np.argmax(score.flatten())
            sampled.append(char_id)
        return sampled
