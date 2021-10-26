import numpy as np
from .abstract import AbstractLayer

class Sigmoid(AbstractLayer):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    def _fwd_prop(self, X):
        out = 1 / (1 + np.exp(-X))
        return  out, out

    def _bckwd_prop(self, X, d_out):
        return X * (1 - X) * d_out, None


class ReLU(AbstractLayer):
    def _fwd_prop(self, X):
        out = np.maximum(X, 0)
        return out, out

    def _bckwd_prop(self, X, d_out):
        grad_in = d_out.copy()
        grad_in[X <= 0] = 0
        return grad_in, None


class Tanh(AbstractLayer):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    def _fwd_prop(self, X):
        out = np.tanh(X)
        return out, out

    def _bckwd_prop(self, X, d_out):
        return (1 - X * X) * d_out, None


class Softmax(AbstractLayer):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    def _fwd_prop(self, X):
        denominator = np.exp(X - np.max(X, axis=1, keepdims=True))
        out = denominator / np.sum(denominator, axis=1, keepdims=True)
        return out, out

    def _bckwd_prop(self, X, d_out):
        dx = X.copy()
        dx[d_out != 0] -= 1
        dx /= d_out.shape[0]
        return dx, None