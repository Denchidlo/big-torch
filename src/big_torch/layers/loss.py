import numpy as np

from ..utils.warnings import warn_on_create
from .abstract import AbstractLayer, layer_registry
from .layer_mixins import DependecyCallMixin


@warn_on_create(
    msg="'CrossEntropy' loss works only with 'softmax' activation layer. Do not even try to use it separately",
    warning_type=RuntimeWarning,
)
@layer_registry.register("multinomial_cross_entropy")
class CrossEntropy(AbstractLayer, DependecyCallMixin):
    output_names = ["y"]

    def __init__(self, shape, softmax_prev=False) -> None:
        super().__init__(shape)

    def _fwd_pass(self, X):
        """
        Warning: Do not call it implicitly

        Params:
            result_tuple - array like collection of (predicted, true results)
        """
        y = X[1]
        y_hat = X[0]

        losses = -np.log(y_hat)
        losses[y == 0] = 0

        return np.sum(losses) / y.shape[0], y_hat

    def _bwd_pass(self, X, d_out):
        """
        Warning: Do not call it implicitly

        Params:
            * X - prediction vector
            * d_out - vector of true results
        """
        return d_out, None


@layer_registry.register("mse")
class MSE(AbstractLayer, DependecyCallMixin):
    output_names = ["y"]

    def __init__(self, shape, softmax_prev=False) -> None:
        super().__init__(shape)

    def _fwd_pass(self, X):
        y = X[1]
        y_hat = X[0]

        return np.mean((y - y_hat) ** 2) / y.shape[0], y_hat

    def _bwd_pass(self, X, d_out):
        return 2 * (X - d_out), None
