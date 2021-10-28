import numpy as np
from .abstract import AbstractLayer, layer_registry


class CrossEntropy(AbstractLayer):
    def __init__(self, shape, softmax_prev=False) -> None:
        super().__init__(shape)

    def _fwd_prop(self, result_tuple):
        """
        Warning: Do not call it implicitly

        Params:
            result_tuple - array like collection of (predicted, true results)
        """
        y = result_tuple[1]
        y_hat = result_tuple[0]

        losses = -np.log(y_hat)
        losses[y == 0] = 0

        return np.sum(losses) / y.shape[0], y_hat

    def _bckwd_prop(self, X, d_out):
        """
        Warning: Do not call it implicitly

        Params:
            * X - prediction vector
            * d_out - vector of true results
        """
        return d_out, None
