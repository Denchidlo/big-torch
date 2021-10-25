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
        out = denominator / np.sum(denominator)
        return out, out

    def _bckwd_prop(self, X, d_out):
        y_hat = X
        diagonalized = np.diagflat(y_hat.flatten())
        sqrd_diagonalized = np.diagflat((y_hat * y_hat).flatten())

        return (y_hat.dot(y_hat.T) - (2*sqrd_diagonalized - diagonalized)).dot(d_out), None