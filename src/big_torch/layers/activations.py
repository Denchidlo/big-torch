import numpy as np

from ..utils.warnings import warn_on_create
from .abstract import AbstractLayer, layer_registry
from .layer_mixins import DependecyCallMixin, GradientStacker1DMixin


@layer_registry.register("sigmoid")
class Sigmoid(AbstractLayer, DependecyCallMixin, GradientStacker1DMixin):
    output_names = ["y"]

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def _fwd_pass(self, X):
        out = 1 / (1 + np.exp(-X))
        return out, out

    def _bwd_pass(self, X, d_out):
        return X * (1 - X) * d_out, None


@layer_registry.register("ReLU")
class ReLU(AbstractLayer, DependecyCallMixin, GradientStacker1DMixin):
    output_names = ["y"]

    def _fwd_pass(self, X):
        out = np.maximum(X, 0)
        return out, out

    def _bwd_pass(self, X, d_out):
        grad_in = d_out.copy()
        grad_in[X <= 0] = 0
        return grad_in, None


@layer_registry.register("tanh")
class Tanh(AbstractLayer, DependecyCallMixin, GradientStacker1DMixin):
    output_names = ["y"]

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def _fwd_pass(self, X):
        out = np.tanh(X)
        return out, out

    def _bwd_pass(self, X, d_out):
        return (1 - X * X) * d_out, None


@warn_on_create(
    msg="'Softmax' activation works only with 'cross-entropy' loss. Do not even try to use it separately",
    warning_type=RuntimeWarning,
)
@layer_registry.register("softmax")
class Softmax(AbstractLayer, DependecyCallMixin, GradientStacker1DMixin):
    output_names = ["y"]

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def _fwd_pass(self, X):
        denominator = np.exp(X - np.max(X, axis=1, keepdims=True))
        out = denominator / np.sum(denominator, axis=1, keepdims=True)
        return out, out

    def _bwd_pass(self, X, d_out):
        return (X - d_out) / d_out.shape[0], None


@layer_registry.register("plain")
class Plain(AbstractLayer, DependecyCallMixin, GradientStacker1DMixin):
    output_names = ["y"]

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def _fwd_pass(self, X):
        return X, X

    def _bwd_pass(self, X, d_out):
        return d_out, None
