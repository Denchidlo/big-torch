import numpy as np
from .abstract import AbstractLayer

class Sigmoid(AbstractLayer):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    def _fwd_prop(self, X):
        out = 1 / (1 + np.exp(-X))
        return  out, out

    def _bckwd_prop(self, X, d_out):
        return X * (X - 1) * d_out, None


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
        denominator = np.exp(X)
        out = denominator / np.sum(denominator, axis=1, keepdims=True)
        return out, out

    def _bckwd_prop(self, X, d_out):
        el_wise = X * d_out
        return (1 - 2*el_wise) + el_wise.sum(axis=1, keepdims=True), None